from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from datasets import load_dataset
import torch
from expertize import ExpertizedLinear
from math import isnan, floor
import gc
from random import shuffle
import itertools as it

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cudnn.enabled = True
torch.backends.cudnn.allow_tf32 = False

model_id = "Meta-Llama-3-8B/"
out_dir = "Meta-Llama-3-8B-MoE/model.pt"

def batched(in_seq, bsz, return_last=True):
    buffer = []
    for item in in_seq:
        buffer.append(item)
        if len(buffer) == bsz:
            yield buffer
            buffer = []
    if return_last and len(buffer) > 0:
        yield buffer
        buffer = []
    return

def set_no_train_mode(module):
    if isinstance(module, ExpertizedLinear):
        module.jitter_noise = 0
        set_req_grad_recursive(module.router_proj, False)
        set_req_grad_recursive(module.experts, False)

def set_router_train_mode(module):
    if isinstance(module, ExpertizedLinear):
        module.jitter_noise = 0
        set_req_grad_recursive(module.router_proj, True)
        set_req_grad_recursive(module.experts, False)

def set_expert_train_mode(module):
    if isinstance(module, ExpertizedLinear):
        module.jitter_noise = 0
        set_req_grad_recursive(module.router_proj, False)
        set_req_grad_recursive(module.experts, True)

def set_dual_train_mode(module):
    if isinstance(module, ExpertizedLinear):
        module.jitter_noise = 0
        set_req_grad_recursive(module, True)

#set requires_grad for all of the parameters of the given module.
def set_req_grad_recursive(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad

#replace a (presumably nn.Linear) module with an ExpertizedLinear module.
def replace_module(module, target_names, new_module_kwargs={}):
    for child_name, child_module in module.named_children():
        if any([x in child_name for x in target_names]):

            module.__setattr__(child_name, ExpertizedLinear.from_linear(child_module, **new_module_kwargs))

            torch.cuda.empty_cache()
            gc.collect()
        else:
            replace_module(child_module, target_names, new_module_kwargs)

#set number of active experts
def set_n_experts(model, n_experts):
    for module in model.modules():
        if isinstance(module, ExpertizedLinear):
            module.top_k = n_experts

#get some samples from the dataset
def fetch_some_samples(dataset, dataset_fraction, tokenizer, max_tokens, seed=42):
    ds_shard = dataset.train_test_split(dataset_fraction, shuffle=True, seed=seed)['test'].map(tokenizer, input_columns='text')

    train_samples = []
    for row in ds_shard:
        num_steps = min([max_tokens, len(row['input_ids'])])
        steps = list(range(2, num_steps))
        train_samples.extend([torch.Tensor([row['input_ids'][:step]]).to(torch.long).cuda() for step in steps])
    
    shuffle(train_samples)

    return train_samples

def train_on_sample():

    router_optim.zero_grad()
    expert_optim.zero_grad()

    minibatch_size = len(input_batch)
    minibatch_losses = []
    for input_sample in input_batch:
        out = model(
            input_ids=input_sample, 
            labels=input_sample,
        )
        loss = out.loss / minibatch_size
        minibatch_losses.append(loss)
        loss.backward()
    #with torch.no_grad():
    #    for param in model.parameters():
    #        if param.requires_grad:
    #            std, mean = torch.std_mean(param.grad)
    #            param.grad = (torch.nn.functional.tanh((param.grad - mean) / std) + mean)
    print(f"Minibatch loss: {sum(minibatch_losses).item()}")
    return sum(minibatch_losses)

def train_dual(lr, weight_decay, dataset, dataset_fraction, max_tokens, seed=42):

    global input_batch, router_optim, expert_optim

    train_samples = fetch_some_samples(dataset=dataset, dataset_fraction=dataset_fraction, tokenizer=tok, max_tokens=max_tokens, seed=seed)

    router_optim = torch.optim.LBFGS([param for name, param in model.named_parameters() if param.requires_grad and "router_proj" in name], lr=0.1)
    expert_optim = torch.optim.NAdam([param for name, param in model.named_parameters() if param.requires_grad and "experts" in name], lr=lr)

    GRAD_ACC_STEPS = 64

    router_scheduler = torch.optim.lr_scheduler.LinearLR(router_optim, start_factor=1, end_factor=0.1, total_iters=4)
    expert_scheduler = torch.optim.lr_scheduler.LinearLR(expert_optim, start_factor=1, end_factor=0.000000001, total_iters=floor(len(train_samples)/GRAD_ACC_STEPS))

    for minibatch in batched(train_samples, GRAD_ACC_STEPS):
        input_batch = minibatch
        router_optim.step(train_on_sample)
        router_scheduler.step()
        print('Trained router.')
        train_on_sample()
        expert_optim.step()
        expert_optim.zero_grad()
        expert_scheduler.step()
        print('Trained experts.')


if __name__ == "__main__":

    MODEL_DIR = "Meta-Llama-3-8B/"

    MODULES_TO_MOE=[
        'gate_proj',
        'up_proj',
        'down_proj',
    ]
    NUM_EXPERTS=16
    ACTIVE_EXPERTS=4


    global model, tok

    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='flash_attention_2')#, quantization_config=qconf)
    set_req_grad_recursive(model, False)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)

    dataset = load_dataset('allenai/c4', 'en.noblocklist', data_files='en.noblocklist/c4-train.00000-of-01024.json.gz', ignore_verifications=True)['train'].select_columns(['text'])

    for i, layer in enumerate(model.model.layers):
        #if i < 7:
        #    continue
#
        replace_module(layer, MODULES_TO_MOE, {'num_experts': NUM_EXPERTS, 'active_experts': ACTIVE_EXPERTS, 'rank_reduction_factor': 1, 'router_bias': False})

        print(f"Training layer {i}'s router")
        layer.apply(set_dual_train_mode)
        train_dual(lr=0.001, weight_decay=5e-3, dataset=dataset, dataset_fraction=0.0002, max_tokens=128, seed=i) #0.001 generally works well
        