"""0002 · The mask and the scale — a reference solution.

Look at this only after you have run the checks at least once. Verify it with

    ./bin/grade 0002 --file exercises/solutions/0002_masking.py
"""

import math

import torch


def causal_mask(T):
    """(T, T): 1 on and below the diagonal (allowed), 0 above it (blocked)."""
    return torch.tril(torch.ones(T, T))


def masked_weights(Q, K, mask):
    """Q, K: (T, dk)   mask: (T, T)   ->   weights: (T, T), rows still sum to 1."""
    dk = Q.shape[-1]                                            # width of one query
    scores = Q @ K.transpose(-2, -1) / math.sqrt(dk)            # score, then scale
    blocked = mask == 0                                         # 0 means "may not look"
    scores = scores.masked_fill(blocked, float("-inf"))          # -inf, not 0
    return torch.softmax(scores, dim=-1)                        # frozen rows still sum to 1


def causal_attention(Q, K, V):
    """Q, K: (T, dk)   V: (T, dv)   ->   output: (T, dv), nothing sees the future."""
    mask = causal_mask(Q.shape[-2])                             # one triangle for all tokens
    return masked_weights(Q, K, mask) @ V                        # (T, T) @ (T, dv)
