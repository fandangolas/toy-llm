import torch
import torch.nn as nn
from torch import Tensor


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"

        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

        # Q, K, V projections in a single matrix multiply
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Causal mask: upper triangle is -inf so future tokens can't be attended to
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape

        # Project and split into Q, K, V
        q, k, v = self.qkv_proj(x).split(self.n_embd, dim=2)

        # Reshape to (B, n_head, T, head_dim)
        def reshape(t: Tensor) -> Tensor:
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # Scaled dot-product attention with causal mask
        scale = self.head_dim ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale                  # (B, n_head, T, T)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)

        out = weights @ v                                            # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)        # (B, T, C)
        return self.resid_drop(self.out_proj(out))
