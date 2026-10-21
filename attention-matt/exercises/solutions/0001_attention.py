"""0001 · Attention is a weighted average — a reference solution.

Look at this only after you have run the checks at least once: the effort of
retrieving the four steps is what makes them stick. Verify it with

    ./bin/grade 0001 --file exercises/solutions/0001_attention.py
"""

import math

import torch


def attention_weights(Q, K):
    """Q: (T, dk)   K: (T, dk)   ->   weights: (T, T), each row summing to 1."""
    dk = Q.shape[-1]                                        # width of one query
    scores = Q @ K.transpose(-2, -1)                        # (T, dk) @ (dk, T) -> (T, T)
    scores = scores / dk ** 0.5                             # the 1/sqrt(dk) scale
    return torch.softmax(scores, dim=-1)                     # each row sums to 1


def attention(Q, K, V):
    """Q, K: (T, dk)   V: (T, dv)   ->   output: (T, dv)."""
    weights = attention_weights(Q, K)                       # (T, T)
    return weights @ V                                      # (T, T) @ (T, dv) -> (T, dv)
