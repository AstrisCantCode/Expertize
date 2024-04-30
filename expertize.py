import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


### Summary:

# LowRankLinear: a module that contains 'a_proj' and 'b_proj' and approximates a linear module (with a low rank).

# ExpertizedLinear: a module that is comprised of a router (module named router_proj) and multiple 'experts'.
#                   Upon initialization, the original weight matrix is decomposed (with SVD) into two weight matrices.
#                   If put directly into a single expert, that expert would be like the original linear layer, 
#                   albeit with computation split up into two matmuls instead of 1. The result is ~the same~, but
#                   there would be more weights (and so, it's slower). To make the model smaller, the rank can be reduced.
#                   This is done simply by truncating the matrices to the desired rank. 
#                   We know this won't *necessarily* be detrimental to model performance, since the LASER paper showed
#                   that high-rank components could actually be *detrimental*. Anyways, next, the matrices are split up
#                   among the experts. By using the router to pick the most important experts, the original 
#                   computation is effectively split up into subproblems, and the most important subproblems are run.



class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=False, dtype=None):
        super().__init__()
        self.a_proj = nn.Linear(in_features, rank, bias=bias, dtype=dtype)
        self.b_proj = nn.Linear(rank, out_features, bias=bias, dtype=dtype)
    def forward(self, x):
        return self.b_proj(self.a_proj(x))


def svd_decompose(in_weights):
    U, S, V = torch.linalg.svd(in_weights.T.to(torch.float), driver='gesvdj', full_matrices=False) 
    weights_a = U
    weights_b = torch.diag(S) @ V
    return weights_a, weights_b

class ExpertizedLinear(nn.Module):
    def __init__(self, in_features, out_features, num_experts, expert_rank, active_experts, router_bias=False, dtype=None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.num_experts = num_experts

        self.router_proj = nn.Linear(in_features, self.num_experts, bias=router_bias, dtype=dtype)
        #nn.init.ones_(self.router_proj.weight.data)

        self.experts = nn.ModuleList([LowRankLinear(in_features, out_features, expert_rank, dtype=dtype) for _ in range(self.num_experts)])

        self.top_k = active_experts

        self.training = True
        self.jitter_noise = 0
    #@torch.compile(mode='max-autotune-no-cudagraphs')
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        """
        forward method from HF Transformers' MixtralSparseMoeBlock class
        """

        batch_size, sequence_length, in_features = hidden_states.shape

        out_features = self.out_features

        #if self.training and self.jitter_noise > 0:
        #    hidden_states = hidden_states * torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, in_features)
        # router_logits: (batch * sequence_length, n_experts)

        router_logits = self.router_proj(F.normalize(hidden_states, dim=-1))
        if self.training and self.jitter_noise > 0:
            temp = router_logits.detach()
            # edit note: noise has been changed from uniformly sampled to sampled from normal distribution, and additive instead of multiplicative. 
            #            consider testing out different distributions, switching addition for multiplication, etc. 
            #            (as an ablation study)
            router_logits = router_logits + torch.normal(mean=0, std=self.jitter_noise*temp.std(dim=0, keepdim=True))

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, out_features), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, in_features)

            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, out_features)
        return final_hidden_states
    
    @classmethod
    def from_linear(cls, linear_module, num_experts, active_experts, rank_reduction_factor=2, router_bias=False): 
        #note: the number of total parameters that the ExpertizedLinear module ends up having is entirely dependent on the
        #      rank reduction factor. 1 or 2 is generally a good starting point. the number of experts can be varied without
        #      significantly changing the number of parameters, up to min(d_in, d_out) (which is generally far higher
        #      than a MoE module should have, eg. experts=8, but min(d_in, d_out)=1024)
        #      For more information, check out this visualization: https://www.desmos.com/calculator/0labfhrnui
        d_out, d_in = linear_module.weight.data.shape
        orig_dtype = linear_module.weight.data.dtype
        orig_device = linear_module.weight.data.device


        expert_rank = int((min([d_in, d_out]) / num_experts) / rank_reduction_factor)

        expertized_linear = ExpertizedLinear(d_in, d_out, num_experts, expert_rank, active_experts, router_bias, dtype=orig_dtype)

        fW_a, fW_b = svd_decompose(linear_module.weight.data)

        del linear_module

        fW_a_splits = fW_a.split(expert_rank, dim=1)
        fW_b_splits = fW_b.split(expert_rank, dim=0)


        for i in range(num_experts):
            expertized_linear.experts[i].a_proj.weight.data = fW_a_splits[i].T.to(orig_dtype)
            expertized_linear.experts[i].b_proj.weight.data = fW_b_splits[i].T.to(orig_dtype)

        return expertized_linear.to(orig_device)

#test_module = nn.Linear(4096, 2048).cuda()
#test_vec = torch.rand((1, 1, 4096)).cuda()
#
#dense_result = test_module(test_vec)
#
#print(dense_result)
#print(dense_result.shape)
#
#test_module_moe = ExpertizedLinear.from_linear(test_module, 4, 2, 1)
#
#sparse_result = test_module_moe(test_vec)
#
#print(sparse_result)
#print(sparse_result.shape)

### THEYRE THE SAME!!!!
### SUCCESS!!!!!

# well, sorta. the router is (was) initialized with ones, so it should assign equal prob
# of each expert for each vector. however, due to some weird floating point stuff (I think),
# its a little inconsistent. As a result, what *should* be [0.5, 0.5] ends up being
# [0.4823, 0.5177]. with finetuning (and hopefully without), this shouldn't affect the model.
