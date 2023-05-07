import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple
# from utils.tensor_functions import compute_in_batches
import sys
import os

from graph_encoder import GraphAttentionEncoder
# from torch.nn import DataParallel
# from utils.beam_search import CachedLookup
# from utils.functions import sample_many

class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key]
        )

class AttentionModel(nn.Module):

    def __init__(self,
                input_dim,
                embedding_dim,
                # hidden_dim,
                n_encode_layers=2,
                tanh_clipping=10.,
                mask_inner=True,
                mask_logits=True,
                normalization='batch',
                n_heads=8,
                # checkpoint_encoder=False,
                # shrink_size=None
                ):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        # self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.tanh_clipping = tanh_clipping
        self.normalization = normalization

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.n_heads = n_heads
        self.temp = 1.0

        self.decode_type = None

        step_context_dim = 2 * embedding_dim  # Embedding of the failiing test and current coverage matrix 
        
        # Learned input symbols for first action
        self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
        self.W_placeholder.data.uniform_(-1, 1)  # normalize initial placeholder

        self.init_embed = nn.Linear(input_dim, embedding_dim)

        self.graph_embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )

        # for attention calculation
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0

        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, input, return_pi=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences
        :return: cost and log_likelihood
        """

        # get embeddings of the test feature (batch_size, graph_size, embed_dim)
        embeddings = self.graph_embedder(self.init_embed(input))

        # get probability and action for each time step (batch_size, time_steps, graph_size)
        _log_p, pi = self._inner(input, embeddings)

        cost, mask = self.problem.get_costs(input, pi)  ## TODO: implement "problem"

        log_likelihood = self._calc_log_likelihood(_log_p, pi, mask)

        return cost, log_likelihood
    
    def _calc_log_likelihood(self, _log_p, a, mask):
        
        # calculate log probability of chosen actions (batch_size, time_steps)
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective
        if mask is not None:
            log_p[mask] = 0
        
        assert (log_p > -1000).data.all()

        return log_p.sum(1)

    def _make_heads(self, v, num_steps=None):
        """
        Use multi-head attention, seperate key, value to get multi-head key, value
        :param v: (batch_size, 1, graph_size, embed_dim)
        :return: (n_heads, batch_size, num_steps, graph_size, head_dim)
        """

        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return(
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )

    def _precompute(self, embeddings, num_steps=1):
        """
        :param embeddings: (batch_size, graph_size, embed_dim)
        """

        # context_embed (batch_size, embed_dim)
        graph_embed = embeddings.mean(1)

        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        # the first part for h_(c)
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # calculate key and value = (batch_size, 1, graph_size, embed_dim) for all node, i.e. formula(5) 
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1) 

        # store precomputed vectors 
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),  
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)
    
    def _get_parallel_step_context(self, embeddings, state):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)
        
        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """

        # (batch_size, num_steps)
        current_node = state.get_current_node()  ## TODO
        batch_size, num_steps = current_node.size()

        if num_steps == 1:
            if state.i.item() == 0:    #### torch.item():get the value of one-element tensor, state.i.item()=0 = has not choose the first step
                return self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1))
            else: # have choose the first step
                return embeddings.gather(
                    1,
                    torch.cat((state.first_a, current_node), 1)[:, :, None].expand(batch_size, 2, embeddings.size(-1)) # (batch_size, 2, embed_dim)
                ).view(batch_size, 1, -1)
        
        # More than one step (batch_size, num_steps - 1, embed_dim)
        embeddings_per_step = embeddings.gather(
            1,
            current_node[:, 1:, None].expand(batch_size, num_steps - 1, embeddings.size(-1))
        )
        return torch.cat(
            (   # return as the embed of the 0 step (batch_size, 1, 2 * embed_dim)
                self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1)),
                torch.cat(
                    (
                        embeddings_per_step[:, 0:1, :].expand(batch_size, num_steps - 1, embeddings.size(-1)), # embed of the first node, expanded as (batch_size, num_steps - 1, embed_dim)
                        embeddings_per_step   # (batch_size, num_steps - 1, embed_dim)
                    ), 2)
            ), 1) # (batch_size, num_steps, embed_dim)

    def _one_to_many_logits(self, query, glimpse_k, glimpse_v, logit_k, mask):
        """
        :query: (batch_size, num_steps, embed_dim)
        :glimpse_k, glimpse_v: (n_heads, batch_size, num_steps, graph_size, head_dim)
        :logit_k: (batch_size, 1, graph_size, embed_dim)

        :return: logits = (batch_size, num_steps, graph_size), glimpse=(batch_size, num_steps, embedding_dim)
        """

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # batch matrix multiplication to compute compatibilities 
        # compatibility = (n_heads, batch_size, num_steps, 1, graph_size)
        compatibility = torch.matmul(glimpse_q, glimpse_k.transpose(-2, -1)) / math.sqrt(glimpse_q.size(-1))
        if self.mask_inner: ## TODO: what is mask inner?
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # compute heads (n_heads, batch_size, num_steps, 1, val_size)
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_v)

        # project to get updated context node embedding (batch_size, num_steps, embedding_dim)
        # after permute: (batch_size, num_steps, 1, n_heads, val_size)
        # get: (batch_size, num_steps, 1, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size)
        )

        # (batch_size, num_steps, 1, embedding_dim)
        final_q = glimpse
        # compute compatibility = 
        # (batch_size, num_steps, 1, embedding_dim) * (batch_size, 1, embed_dim, graph_size) =
        # (batch_size, num_steps, 1, graph_size)
        logits = torch.matmul(final_q, logit_k.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_q.size(-1))

        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf
        
        return logits, glimpse.squeeze(-2)


    def _get_log_p(self, fixed, state, normalize=True):

        # compute query = context node embedding
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state)) ## TODO
        
        # compute keys and values for the nodes. glimpse_k, glimpse_v = (n_heads, batch_size, num_steps, graph_size, head_dim)
        glimpse_k, glimpse_v, logit_k = fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

        # compute the mask
        mask = state.get_mask()

        # compute logits (unnormalized log_p)
        # log_p = (batch_size, num_steps, graph_size)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_k, glimpse_v, logit_k, mask)

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, mask

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:
            self.temp = temp

    def _select_node(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(-1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1) #采样的时候是根据输入张量的数值当做权重来进行抽样的

            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print("Sampled bad values, resampling!")
                selected = probs.multinomial(1).squeeze(1)
        else:
            assert False, "Unknown decode type"

        return selected


    # calculate attention and get policy
    def _inner(self, input, embeddings):
        
        outputs = []
        sequences = []

        state = self.problem.make_state(input) # TODO: implement state

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self._precompute(embeddings)

        batch_size = input.size(0)

        # Perform decoding steps (select 10 test cases for each buggy version)
        iter = 10
        for _ in range(iter):
            # for each time step, get the probability and mask, (batch_size, graph_size)
            log_p, mask = self._get_log_p(fixed, state)  

            # select the next node
            selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :]) # squeeze out steps dimension

            state = state.update(selected)

            outputs.append(log_p[:, 0, :])
            sequences.append(selected)
        
        return torch.stack(outputs, 1), torch.stack(sequences, 1)