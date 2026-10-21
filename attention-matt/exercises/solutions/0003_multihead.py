"""0003 · Several weighted averages at once — a reference solution.

Look at this only after you have run the checks at least once. Verify it with

    ./bin/grade 0003 --file exercises/solutions/0003_multihead.py
"""

import math

import torch


def split_heads(x, n_head):
    """(T, C) -> (H, T, dh): the width divides in place, then heads move forward."""
    T, C = x.shape
    dh = C // n_head
    return x.view(T, n_head, dh).transpose(0, 1)     # (T, H, dh) -> (H, T, dh)


def merge_heads(x):
    """(H, T, dh) -> (T, H * dh): the inverse of split_heads."""
    n_head, T, dh = x.shape
    return x.transpose(0, 1).reshape(T, n_head * dh)  # reshape, not view: it is transposed


def head_weights(Q, K, n_head):
    """(T, C), (T, C) -> (H, T, T): lesson 02's causal weights, once per head."""
    Qh = split_heads(Q, n_head)                      # (H, T, dh)
    Kh = split_heads(K, n_head)                      # (H, T, dh)
    dk = Qh.shape[-1]                                # the head width, not C
    scores = Qh @ Kh.transpose(-2, -1) / math.sqrt(dk)   # (H, T, T)
    T = Qh.shape[-2]
    causal = torch.tril(torch.ones(T, T))            # one mask, shared by all heads
    return torch.softmax(scores.masked_fill(causal == 0, float("-inf")), dim=-1)


def multi_head_attention(Q, K, V, n_head):
    """(T, C) x3 -> (T, C): H weighted averages, concatenated."""
    weights = head_weights(Q, K, n_head)             # (H, T, T)
    mixed = weights @ split_heads(V, n_head)         # (H, T, T) @ (H, T, dh) -> (H, T, dh)
    return merge_heads(mixed)                        # (T, C)
