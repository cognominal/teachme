"""0003 · Several weighted averages at once — your implementation.

Four functions, then run the checks:

    ./bin/grade 0003

or, to see your own numbers as a table in the terminal:

    ./.venv/bin/python exercises/0003_multihead.py

Lesson 02 gave you one weighted average per token. This lesson runs H of them
in parallel over slices of the width, and concatenates the results. Nothing
about the attention itself changes — the score matrix, the scale and the mask
are exactly what you already wrote. What changes is the shape bookkeeping that
puts H (T × T) matrices in one tensor, and takes them apart again.
"""

import math

import torch


def split_heads(x, n_head):
    """x: (T, C)   ->   heads: (n_head, T, C // n_head).

    The width C is *divided*, not copied: head h gets columns
    h * dh … (h + 1) * dh - 1, where dh = C // n_head.

    The idiom, as it appears in real model code:
        x.view(T, n_head, dh)        -> (T, H, dh): the width splits in place
        .transpose(0, 1)             -> (H, T, dh): heads become a leading axis
    """
    raise NotImplementedError("split_heads is not written yet")


def merge_heads(x):
    """x: (n_head, T, dh)   ->   (T, n_head * dh).

    The inverse of split_heads: put the heads back side by side, in order, so
    that column h * dh of the result is column 0 of head h.

    You will need to transpose back to (T, H, dh) and then flatten the last two
    axes. If torch refuses to .view() a transposed tensor — it will, sometimes,
    with "view size is not compatible with input tensor's size and stride" —
    that is exactly what .reshape() is for.
    """
    raise NotImplementedError("merge_heads is not written yet")


def head_weights(Q, K, n_head):
    """Q, K: (T, C)   ->   weights: (n_head, T, T), one causal matrix per head.

    Per head h, this is lesson 02 exactly, with dk = C // n_head:
        scores = Qh @ Kh.transpose(-2, -1)   -> (H, T, T)
        scores = scores / sqrt(dk)
        scores = scores.masked_fill(causal mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
    The mask is shared by every head — it says what is allowed, not what is
    attended to.
    """
    raise NotImplementedError("head_weights is not written yet")


def multi_head_attention(Q, K, V, n_head):
    """Q, K, V: (T, C)   ->   output: (T, C).

    Three lines on top of what you already have: split V, mix it with the
    per-head weights, merge the heads back together.
    """
    raise NotImplementedError("multi_head_attention is not written yet")


# ---------------------------------------------------------------------------
# The example the lesson draws: the same 4 tokens as lessons 01 and 02, but
# with width C = 4, so two heads of width 2. Head 0's slice is *exactly* the
# Q and K of the earlier lessons, so its weights should reproduce lesson 02's
# table here.
#
# Running this file prints the split, both heads' weights, the merged output,
# and the properties that must hold if the heads are doing their job.

EXAMPLE_TOKENS = ["the", "trophy", "fit", "it"]
EXAMPLE_N_HEAD = 2
EXAMPLE_Q = torch.tensor([[0.2, 0.9, 0.1, 0.4],
                          [0.9, 0.2, 0.5, 0.1],
                          [0.6, 0.4, 0.7, 0.2],
                          [1.0, 0.0, 0.3, 0.8]])
EXAMPLE_K = torch.tensor([[0.2, 0.9, 0.3, 0.2],
                          [1.0, 0.1, 0.4, 0.6],
                          [0.6, 0.5, 0.2, 0.7],
                          [0.3, 0.7, 0.9, 0.1]])
EXAMPLE_V = torch.tensor([[0.1, 0.1, 0.1, 0.9],
                          [1.0, 0.2, 0.0, 0.3],
                          [0.0, 0.6, 0.4, 0.2],
                          [0.2, 0.2, 0.6, 0.5]])


def show():
    """Print the split, both heads, and the merged output for the example."""
    torch.set_printoptions(precision=3, sci_mode=False)
    n_head = EXAMPLE_N_HEAD
    heads = split_heads(EXAMPLE_Q, n_head)
    weights = head_weights(EXAMPLE_Q, EXAMPLE_K, n_head)
    output = multi_head_attention(EXAMPLE_Q, EXAMPLE_K, EXAMPLE_V, n_head)

    dh = EXAMPLE_Q.shape[-1] // n_head
    print(f"the width is divided: C = {EXAMPLE_Q.shape[-1]}, H = {n_head}, dh = {dh}")
    print(f"Q split into heads: {tuple(heads.shape)} — head 0 = columns 0..{dh - 1}, "
          f"head 1 = columns {dh}..{2 * dh - 1}")
    for h in range(n_head):
        print(f"\nhead {h} queries (same tokens, different slice of the width)")
        for token, row in zip(EXAMPLE_TOKENS, heads[h].tolist()):
            print(f"{token:>7} " + "  ".join(f"{v:7.2f}" for v in row))

    for h in range(n_head):
        print(f"\nhead {h} weights — row = looking, column = looked at (causal)")
        print("        " + "  ".join(f"{t:>7}" for t in EXAMPLE_TOKENS))
        for token, row in zip(EXAMPLE_TOKENS, weights[h].tolist()):
            print(f"{token:>7} " + "  ".join(f"{w:7.3f}" for w in row))

    print("\noutput — the heads, concatenated back to width", EXAMPLE_Q.shape[-1])
    print(output)

    print("\nproperties that must hold:")
    print("  row 0 of the output is exactly V[0] (each head sees only itself):",
          torch.equal(output[0], EXAMPLE_V[0]))

    blocked = (weights == 0).sum().item()
    print(f"  blocked cells are exactly 0 ({blocked} of them across {n_head} heads):",
          blocked == n_head * (len(EXAMPLE_TOKENS) * (len(EXAMPLE_TOKENS) - 1) // 2))

    # One head must behave exactly like the attention you already wrote.
    one = multi_head_attention(EXAMPLE_Q, EXAMPLE_K, EXAMPLE_V, 1)
    dk = EXAMPLE_Q.shape[-1]
    mask = torch.tril(torch.ones(len(EXAMPLE_TOKENS), len(EXAMPLE_TOKENS)))
    scores = EXAMPLE_Q @ EXAMPLE_K.transpose(-2, -1) / math.sqrt(dk)
    plain_causal = torch.softmax(scores.masked_fill(mask == 0, float("-inf")), dim=-1) @ EXAMPLE_V
    gap = (one - plain_causal).abs().max().item()
    print(f"  with n_head = 1 the output is lesson 02's attention again "
          f"(largest difference {gap:.2e})")

    # Heads do not mix on their own: head 0's output columns cannot notice a
    # change in head 1's slice of V.
    tampered_v = EXAMPLE_V.clone()
    tampered_v[:, dh:] = 99.0
    other = multi_head_attention(EXAMPLE_Q, EXAMPLE_K, tampered_v, n_head)
    print("  head 0's output columns are unchanged when head 1's values change:",
          torch.equal(other[:, :dh], output[:, :dh]))

    # And splitting is not a no-op: one 4-wide head is a different function.
    gap = (one - output).abs().max().item()
    print(f"  one {EXAMPLE_Q.shape[-1]}-wide head vs {n_head} of width {dh}: "
          f"the outputs differ by {gap:.4f}")


if __name__ == "__main__":
    show()
