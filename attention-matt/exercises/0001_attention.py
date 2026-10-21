"""0001 · Attention is a weighted average — your implementation.

Write the two functions, then run the checks:

    ./bin/grade 0001

or, to see your own numbers as a table in the terminal:

    ./.venv/bin/python exercises/0001_attention.py

The checks compare your output against an independent reference implementation
on random tensors, to 1e-5. `torch` is imported for you; nothing else is needed.
"""

import math

import torch


def attention_weights(Q, K):
    """Q: (T, dk)   K: (T, dk)   ->   weights: (T, T), each row summing to 1.

    Steps, in order:
      1. score   — every query against every key      -> (T, T)
      2. scale   — divide by the square root of dk
      3. normalise — each row becomes a distribution over the keys
    """
    raise NotImplementedError("attention_weights is not written yet")


def attention(Q, K, V):
    """Q, K: (T, dk)   V: (T, dv)   ->   output: (T, dv).

    One line, once attention_weights exists: mix the rows of V by the weights.
    """
    raise NotImplementedError("attention is not written yet")


# ---------------------------------------------------------------------------
# The example the lesson draws: 4 tokens, query width dk = 2, value width dv = 3.
# Running this file directly prints your own weights and output, so you get a
# second feedback loop that needs no grader at all.

EXAMPLE_TOKENS = ["the", "trophy", "fit", "it"]
EXAMPLE_Q = torch.tensor([[0.2, 0.9], [0.9, 0.2], [0.6, 0.4], [1.0, 0.0]])
EXAMPLE_K = torch.tensor([[0.2, 0.9], [1.0, 0.1], [0.6, 0.5], [0.3, 0.7]])
EXAMPLE_V = torch.tensor([[0.1, 0.1, 0.1], [1.0, 0.2, 0.0], [0.0, 0.6, 0.4], [0.2, 0.2, 0.6]])


def show():
    """Print the weights and the output for the lesson's example."""
    torch.set_printoptions(precision=3, sci_mode=False)
    weights = attention_weights(EXAMPLE_Q, EXAMPLE_K)
    output = attention(EXAMPLE_Q, EXAMPLE_K, EXAMPLE_V)

    print("weights — row = the token looking, column = the token looked at")
    print("        " + "  ".join(f"{t:>7}" for t in EXAMPLE_TOKENS))
    for token, row in zip(EXAMPLE_TOKENS, weights.tolist()):
        print(f"{token:>7} " + "  ".join(f"{w:7.3f}" for w in row))

    print("\nrow sums:", [round(s, 6) for s in weights.sum(dim=-1).tolist()])

    print("\noutput — each token's new vector, width dv = 3")
    print(output)


if __name__ == "__main__":
    show()
