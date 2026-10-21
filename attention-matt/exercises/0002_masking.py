"""0002 · The mask and the scale — your implementation.

Three functions, then run the checks:

    ./bin/grade 0002

or, to see your own numbers as a table in the terminal:

    ./.venv/bin/python exercises/0002_masking.py

Everything from lesson 01 is still true — score, scale, softmax each row, mix
the rows of V — plus one new step. The mask is not a new mechanism bolted on
the side: it is an edit to the score matrix, in the one place that decides
everything the layer is allowed to do.
"""

import math

import torch


def causal_mask(T):
    """T -> (T, T) tensor: 1 where a query may look, 0 where it may not.

    Causal means position i may look at 0, 1, … i and nothing after it.
    torch.tril(torch.ones(T, T)) is exactly that triangle, diagonal included —
    a token is allowed to see itself.
    """
    raise NotImplementedError("causal_mask is not written yet")


def masked_weights(Q, K, mask):
    """Q: (T, dk)   K: (T, dk)   mask: (T, T) of 0s and 1s   ->   weights: (T, T).

    The five steps, in order — step 3 is the new one:
      1. scores  — Q @ K.transpose(-2, -1)          -> (T, T)
      2. scale   — scores / dk ** 0.5
      3. block   — where the mask is 0, make the score float("-inf")
      4. softmax — dim=-1, along each row
      5. (done)  — each row is now a distribution over the keys it may see
    """
    raise NotImplementedError("masked_weights is not written yet")


def causal_attention(Q, K, V):
    """Q, K: (T, dk)   V: (T, dv)   ->   output: (T, dv).

    Two lines, built from the two functions above: no token may see the future.
    """
    raise NotImplementedError("causal_attention is not written yet")


# ---------------------------------------------------------------------------
# The same 4 tokens as lesson 01, so you can compare the two pages directly.
# Running this file prints the mask, your masked weights, your output, and two
# properties that must hold if the mask is doing its job.

EXAMPLE_TOKENS = ["the", "trophy", "fit", "it"]
EXAMPLE_Q = torch.tensor([[0.2, 0.9], [0.9, 0.2], [0.6, 0.4], [1.0, 0.0]])
EXAMPLE_K = torch.tensor([[0.2, 0.9], [1.0, 0.1], [0.6, 0.5], [0.3, 0.7]])
EXAMPLE_V = torch.tensor([[0.1, 0.1, 0.1], [1.0, 0.2, 0.0], [0.0, 0.6, 0.4], [0.2, 0.2, 0.6]])


def show():
    """Print the mask, the weights and the output for the example above."""
    torch.set_printoptions(precision=3, sci_mode=False)
    mask = causal_mask(len(EXAMPLE_TOKENS))
    weights = masked_weights(EXAMPLE_Q, EXAMPLE_K, mask)
    output = causal_attention(EXAMPLE_Q, EXAMPLE_K, EXAMPLE_V)

    print("mask — 1 = this query may look at that key")
    print("        " + "  ".join(f"{t:>7}" for t in EXAMPLE_TOKENS))
    for token, row in zip(EXAMPLE_TOKENS, mask.tolist()):
        print(f"{token:>7} " + "  ".join(f"{int(v):7d}" for v in row))

    print("\nweights — a blocked cell is exactly 0, not nearly 0")
    print("        " + "  ".join(f"{t:>7}" for t in EXAMPLE_TOKENS))
    for token, row in zip(EXAMPLE_TOKENS, weights.tolist()):
        print(f"{token:>7} " + "  ".join(f"{w:7.3f}" for w in row))

    print("\nrow sums:", [round(s, 6) for s in weights.sum(dim=-1).tolist()])

    print("\noutput — each token's new vector, width dv = 3")
    print(output)

    print("\ntwo properties that must hold:")
    print("  row 0 is exactly V[0] (only itself was visible):",
          torch.equal(output[0], EXAMPLE_V[0]))

    # The future cannot leak. Replace the last value row with something absurd:
    # if the mask works, the output rows before it cannot move at all.
    tampered_v = EXAMPLE_V.clone()
    tampered_v[3] = torch.tensor([99.0, -99.0, 42.0])
    tampered = causal_attention(EXAMPLE_Q, EXAMPLE_K, tampered_v)
    print("  rows 0-2 unchanged when token 3's value is replaced:",
          torch.equal(tampered[:3], output[:3]))


if __name__ == "__main__":
    show()
