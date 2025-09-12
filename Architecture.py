import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm.auto import tqdm
from contextlib import nullcontext
import os

import numpy as np


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
        