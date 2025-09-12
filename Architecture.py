import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm.auto import tqdm
from contextlib import nullcontext
import os

import numpy as np

# Congif Gemma 3 
GEMMA3_CONFIG_270M = {
    "vocab_size": 50257,
    "context_length": 32_768,
    "emb_dim": 640,
    "n_heads": 4,
    "n_layers": 18,
    "hidden_dim": 2048,
    "head_dim": 256,
    "qk_norm": True,
    "n_kv_groups": 1,
    "rope_local_base": 10_000.0,
    "rope_base": 1_000_000.0,
    "sliding_window": 512,
      "layer_types": [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention"
    ],
    "dtype": torch.bfloat16,
    "query_pre_attn_scalar": 256,
}

# Rotary Positional Embeddings (RoPE) functions
def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype= torch.float32):
    """
    Compute the RoPE (Rotary Positional Embedding) parameters.

    Args:
        head_dim (int): The dimension of each attention head.
        theta_base (int, optional): The base value for theta. Default is 10,000.
        context_length (int, optional): The maximum context length. Default is 4096.
        dtype (torch.dtype, optional): The data type for the resulting tensor. Default is torch.float32.

    Returns:
        torch.Tensor: A tensor of shape (head_dim // 2,) containing the RoPE parameters.
    """

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # Generate position indices
    position = torch.arange(context_length, dtype=dtype)

    # Compute the angles 
    angles = position[:, None] * inv_freq[None, :] # Shape: (context_length, head_dim // 2)

    angles = torch.cat([angles, angles], dim=1) # Shape: (context_length, head_dim)


    # Compute the RoPE parameters
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin

def apply_rope(x, cos, sin): 
    # x: (batch_size, num_heads, seq_len, head_dim)
    Batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, 'Head dimension must be even'

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2] # First half
    x2 = x[..., head_dim // 2 :] # Second half

    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0) # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0) # Shape: (1, 1, seq_len, head_dim)

    rotated = torch.cat((-x2, x1), dim=-1) # Shape: (batch_size, num_heads, seq_len, head_dim)
    x_rotated = (x * cos) + (rotated * sin) # Shape: (batch_size, num_heads, seq_len, head_dim)

    return x_rotated.to(dtype=x.dtype) 


# Model architecture
class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-8, bias=False, **kwargs):
        super().__init__()
        self.eps = eps
        # Gemma3 stores zero-centered weights and uses (1 + weight) during forward
        self.scale = nn.Parameter(torch.zeros(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype
        x_f = x.float()
        var = x_f.pow(2).mean(-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(var + self.eps)
        out = x_norm * (1.0 + self.scale.float())

        if self.shift is not None: 
            out = out + self.shift.float()

        return out.to(input_dtype)

class GroupedQueryAttention(nn.Module): 
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, query_pre_attn_scalar=None, dtype=None):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads 
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None: 
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim
        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm: 
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else: 
            self.q_norm = self.k_norm = None
        
        if query_pre_attn_scalar is not None:
            self.scaling = (query_pre_attn_scalar) ** -0.5
        else: 
            self.scaling = head_dim ** -0.5

    def forward(self, x, mask, cos, sin):
        b, num_tokens, _ = x.shape()

        # APply projections
        queries = self.W_query(x) # (b, num_tokens, num_heads * head_dim)
        keys = self.W_key(x)     # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x) # (b, num_tokens, num_kv_groups * head_dim)

        # Reshape 
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2) # (b, num_heads, num_tokens, head_dim)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)       # (b, num_kv_groups, num_tokens, head_dim)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)   # (b, num_kv_groups, num_tokens, head_dim)

        # Optional normalization
        if self.q_norm: 
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        # Apply RoPE
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        # Expand K and V to match number of heads 
        keys = keys.repeat_interleave(self.group_size, dim=1)   # (b, num_heads, num_tokens, head_dim)
        values = values.repeat_interleave(self.group_size, dim=1) # (b, num_heads, num_tokens, head_dim)

        # Scale queries 
        queries = queries * self.scaling

        # Attention
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        context = (attn_weights @ values).tranpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context)
    

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.gelu(x_fc1, approximate="tanh") * x_fc2
        return self.fc3(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: dict, attn_type: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.attn_type = attn_type
        self.att = GroupedQueryAttention(
            d_in=cfg['emb_dim'],
            num_heads=cfg['n_heads'],
            num_kv_groups=cfg['n_kv_groups'],
            head_dim=cfg['head_dim'],
            qk_norm=cfg['qk_norm'],
            query_pre_attn_scalar=cfg['query_pre_attn_scalar'],
            dtype=cfg['dtype'],
        )

        self.ff = FeedForward(cfg)
        self.input_layernorm = RMSNorm(cfg['emb_dim'], eps=1e-6)
        self.post_attention_layer_norm = RMSNorm(cfg['emb_dim'], eps=1e-6)
        self.pre_feedforward_layer_norm = RMSNorm(cfg['emb_dim'], eps=1e-6)
        self.post_feedforward_layer_norm = RMSNorm(cfg['emb_dim'], eps=1e-6)

    def forward(
            self,
            x, 
            mask_global, 
            mask_local,
            cos_global,
            sin_global,
            cos_local,
            sin_local
    ):
        # Shortcut connection for attention block
        shortcut = x

        # Passing to the first layer norm
        x = self.input_layernorm(x)

        # Passing to attention block
        if self.attn_type == "sliding_attention":
            attn_mask = mask_local
            cos = cos_local
            sin = sin_local
        else:
            attn_mask = mask_global
            cos = cos_global
            sin = sin_global

        x_attn = self.att(x, attn_mask, cos, sin)
        x_attn = self.post_attention_layer_norm(x_attn)
        x = shortcut + x_attn

        # After the attention block -> Passing to a feed forward network 
        # Shortcut connection for feedforward block

        shortcut = x
        x_ffn = self.pre_feedforward_layer_norm(x)
        x_ffn = self.ff(x_ffn)
        x_ffn = self.post_feedforward_layer_norm(x_ffn)
        x = shortcut + x_ffn

        return x
    
    # Entire Gemma 3 Architecture
    class Gemma3Model(nn.Module):
        def __init__(self,cfg: dict,  *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert cfg['layer_types'] is not None and len(cfg['layer_types']) == cfg['n_layers'], "cfg['layer_types'] must be a list of length cfg['n_layers']"

            # Main Model Parameters 
            self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'], dtype=cfg['dtype'])
            self.blocks = nn.ModuleList([
                TransformerBlock(cfg, attn_type) for attn_type in cfg['layer_types']
            ])

            self.final_layer_norm = RMSNorm(cfg['emb_dim'], eps=1e-6)
            self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False, dtype=cfg['dtype'])
            self.cfg = cfg

            
