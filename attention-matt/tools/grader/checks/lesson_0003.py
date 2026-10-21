"""lesson_0003.py — checks for "Several weighted averages at once".

The references here are deliberately written with a *different* idiom from the
one the lesson teaches: the checks split heads by slicing and stack them, where
the solution reshapes and transposes. Same numbers, independent construction.

The last part is checked against `F.scaled_dot_product_attention` as well, so
the learner's five lines per head are compared against the fused kernel.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .support import ATOL, as_tensor, match_detail, near, randn, shape_of

# ---------------------------------------------------------------- references


def reference_split(x: torch.Tensor, n_head: int) -> torch.Tensor:
    """(T, C) -> (H, T, dh), by slicing the width and stacking the slices."""
    _, C = x.shape
    dh = C // n_head
    return torch.stack([x[:, h * dh:(h + 1) * dh] for h in range(n_head)])


def reference_merge(x: torch.Tensor) -> torch.Tensor:
    """(H, T, dh) -> (T, H * dh), by concatenating the heads."""
    return torch.cat([x[h] for h in range(x.shape[0])], dim=-1)


def reference_head_weights(Q: torch.Tensor, K: torch.Tensor, n_head: int) -> torch.Tensor:
    Qh, Kh = reference_split(Q, n_head), reference_split(K, n_head)
    dh = Qh.shape[-1]
    scores = Qh @ Kh.transpose(-2, -1) / math.sqrt(dh)
    T = Qh.shape[-2]
    causal = torch.tril(torch.ones(T, T))
    return torch.softmax(scores.masked_fill(causal == 0, float("-inf")), dim=-1)


def reference_multi_head(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n_head: int) -> torch.Tensor:
    weights = reference_head_weights(Q, K, n_head)
    return reference_merge(weights @ reference_split(V, n_head))


def fused_multi_head(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n_head: int) -> torch.Tensor:
    """The production path: one fused kernel call per head, then concatenate."""
    Qh, Kh, Vh = reference_split(Q, n_head), reference_split(K, n_head), reference_split(V, n_head)
    heads = [F.scaled_dot_product_attention(Qh[h], Kh[h], Vh[h], is_causal=True) for h in range(n_head)]
    return torch.cat(heads, dim=-1)


def reference_multi_head_upper(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n_head: int) -> torch.Tensor:
    """The same block with the mask upside down — a reference for the diagnosis."""
    Qh, Kh, Vh = reference_split(Q, n_head), reference_split(K, n_head), reference_split(V, n_head)
    T = Qh.shape[-2]
    upper = torch.triu(torch.ones(T, T))
    scores = Qh @ Kh.transpose(-2, -1) / math.sqrt(Qh.shape[-1])
    weights = torch.softmax(scores.masked_fill(upper == 0, float("-inf")), dim=-1)
    return reference_merge(weights @ Vh)


def reference_multi_head_reversed_heads(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n_head: int) -> torch.Tensor:
    """The same block with the heads concatenated back-to-front — a diagnosis reference."""
    weights = reference_head_weights(Q, K, n_head)
    mixed = weights @ reference_split(V, n_head)
    return reference_merge(mixed.flip(0))


def plain_causal_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Lesson 02's answer, on the full width — what n_head = 1 has to reproduce."""
    scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])
    T = Q.shape[-2]
    causal = torch.tril(torch.ones(T, T))
    return torch.softmax(scores.masked_fill(causal == 0, float("-inf")), dim=-1) @ V


# Probe tensors: T ≠ H, so that a missing or wrong transpose changes the shape
# (and so a swapped pair of axes cannot hide behind a coincidence).
P_T, P_C, P_H = 6, 8, 2
P_DH = P_C // P_H
P_X = randn(P_T, P_C, seed=200)
P_Q = randn(P_T, P_C, seed=201)
P_K = randn(P_T, P_C, seed=202)
P_V = randn(P_T, P_C, seed=203)
P_HEADS = randn(P_H, P_T, P_DH, seed=205)


# --------------------------------------------------------------------- Part A

PART_A_TESTS = [
    {
        "name": "returns (n_head × T × dh)",
        "run": lambda fn: _shape_is(fn(randn(5, 6, seed=11), 3), (3, 5, 2)),
    },
    {
        "name": "head h is columns h·dh to (h+1)·dh of the input",
        "run": lambda fn: _per_head_slice(fn),
    },
    {
        "name": "matches an independent split on random input",
        "run": lambda fn: match_detail(fn(P_X, P_H), reference_split(P_X, P_H)),
    },
    {
        "name": "one head keeps the whole width",
        "run": lambda fn: match_detail(fn(P_X, 1), reference_split(P_X, 1)),
    },
    {
        "name": "as many heads as columns gives one number per token per head",
        "run": lambda fn: match_detail(fn(randn(4, 4, seed=12), 4), reference_split(randn(4, 4, seed=12), 4)),
    },
]

PART_A_DIAGNOSE = [
    {
        "when": lambda fn: _shape_is_raw(fn, (P_T, P_H, P_DH), (P_X, P_H)),
        "hint": "You split the width but never moved the head axis to the front: the result is still (T, H, dh). Reaching into head h needs <code>[h]</code> to be the head — append <code>.transpose(0, 1)</code>.",
    },
    {
        "when": lambda fn: near(fn(P_X, P_H), P_X.reshape(P_T, P_H, P_DH)),
        "hint": "The heads are read the wrong way round: you have the width grouped as (T, H, dh) without the transpose, so <code>heads[t]</code> is a token rather than a head. <code>x.view(T, H, dh).transpose(0, 1)</code>.",
    },
    {
        "when": lambda fn: near(fn(P_X, P_H), P_X.view(P_H, P_T, P_DH)),
        "hint": "The axes are swapped: for a (T, C) input the first axis is tokens, so <code>view(H, T, dh)</code> re-reads rows as heads. Split the *last* axis first (<code>view(T, H, dh)</code>) and then transpose the first two.",
    },
    {
        "when": lambda fn: True,
        "hint": "Two steps: <code>dh = x.shape[-1] // n_head</code>, then <code>x.view(x.shape[0], n_head, dh).transpose(0, 1)</code>. Print the shape after each step — <code>(T, H, dh)</code> then <code>(H, T, dh)</code>.",
    },
]

# --------------------------------------------------------------------- Part B

PART_B_TESTS = [
    {
        "name": "returns (T × H·dh)",
        "run": lambda fn: _shape_is(fn(P_HEADS), (P_T, P_C)),
    },
    {
        "name": "matches an independent merge on random input",
        "run": lambda fn: match_detail(fn(P_HEADS), reference_merge(P_HEADS)),
    },
    {
        "name": "undoes split_heads exactly",
        "run": lambda fn: _round_trip(fn),
    },
    {
        "name": "head h lands in columns h·dh to (h+1)·dh",
        "run": lambda fn: _ordered_columns(fn),
    },
    {
        "name": "one head passes straight through",
        "run": lambda fn: match_detail(fn(randn(1, 5, 3, seed=21)), randn(5, 3, seed=21)),
    },
]

PART_B_DIAGNOSE = [
    {
        "when": lambda fn: _raises_view_error(fn),
        "hint": "torch refused <code>.view()</code> on a transposed tensor — that is the error above, and it is the reason <code>.reshape()</code> exists. <code>x.transpose(0, 1).reshape(T, H * dh)</code>, or <code>x.transpose(0, 1).contiguous().view(T, H * dh)</code>.",
    },
    {
        "when": lambda fn: _shape_is_raw(fn, (P_T, P_H, P_DH), (P_HEADS,)),
        "hint": "You transposed the heads to the end but never flattened the last two axes: the result is (T, H, dh). One more step — <code>.reshape(T, H * dh)</code> — puts the heads side by side.",
    },
    {
        "when": lambda fn: near(fn(P_HEADS), reference_merge(P_HEADS.flip(0))),
        "hint": "The heads are concatenated in the wrong order — the last head's numbers land in the first columns. <code>torch.cat</code>/<code>reshape</code> walk head 0 first; do not reverse anything.",
    },
    {
        "when": lambda fn: True,
        "hint": "It is the inverse of Part A: <code>x.transpose(0, 1)</code> puts the token axis back in front, then <code>.reshape(T, H * dh)</code> flattens the head and width axes into one column range.",
    },
]

# --------------------------------------------------------------------- Part C

PART_C_TESTS = [
    {
        "name": "returns (H × T × T)",
        "run": lambda fn: _shape_is(fn(randn(5, 6, seed=31), randn(5, 6, seed=32), 2), (2, 5, 5)),
    },
    {
        "name": "matches an independent implementation on random input",
        "run": lambda fn: match_detail(fn(P_Q, P_K, P_H), reference_head_weights(P_Q, P_K, P_H)),
    },
    {
        "name": "every row of every head sums to 1",
        "run": lambda fn: _rows_sum_to_one(fn),
    },
    {
        "name": "blocked cells are exactly 0 in every head",
        "run": lambda fn: _blocked_are_zero(fn),
    },
    {
        "name": "each head reads only its own slice of K",
        "run": lambda fn: _heads_read_their_own_slice(fn),
    },
    {
        "name": "one head is lesson 02's attention on the full width",
        "run": lambda fn: _one_head_is_plain(fn),
    },
]

PART_C_DIAGNOSE = [
    {
        "when": lambda fn: near(fn(P_Q, P_K, P_H), torch.softmax(reference_split(P_Q, P_H) @ reference_split(P_K, P_H).transpose(-2, -1) / math.sqrt(P_DH), dim=-1)),
        "hint": "These are unmasked per-head weights — the heads are right, the causal mask is missing. Lesson 02's <code>masked_fill(..., float(\"-inf\"))</code> still applies, once per head.",
    },
    {
        "when": lambda fn: near(fn(P_Q, P_K, P_H), torch.softmax((reference_split(P_Q, P_H) @ reference_split(P_K, P_H).transpose(-2, -1) / math.sqrt(P_DH)).masked_fill(torch.tril(torch.ones(P_T, P_T)) == 0, 0.0), dim=-1)),
        "hint": "The mask is there, but the blocked scores were filled with <code>0</code> instead of −∞ — <code>exp(0) = 1</code> hands every blocked token a share (lesson 02's trap, now repeated per head).",
    },
    {
        "when": lambda fn: near(fn(P_Q, P_K, P_H), torch.softmax((reference_split(P_Q, P_H) @ reference_split(P_K, P_H).transpose(-2, -1)).masked_fill(torch.tril(torch.ones(P_T, P_T)) == 0, float("-inf")), dim=-1)),
        "hint": "No <code>1/√dh</code> scale: the divisor is the <em>head</em> width (C // n_head), not the full width C — and it certainly is not 1. Without it every head saturates.",
    },
    {
        "when": lambda fn: near(fn(P_Q, P_K, P_H), reference_head_weights(P_Q, P_K, P_H).transpose(1, 2)),
        "hint": "The heads are right but transposed: the result is (H, T, T) with the last two axes swapped, so <code>[h, i, j]</code> is key <em>i</em> attending to query <em>j</em>. The mask should be the lower triangle of <code>[h, i, j]</code>.",
    },
    {
        "when": lambda fn: near(fn(P_Q, P_K, P_H), torch.softmax((reference_split(P_Q, P_H) @ reference_split(P_K, P_H).transpose(-2, -1) / math.sqrt(P_DH)).masked_fill(torch.tril(torch.ones(P_T, P_T)) == 0, float("-inf")), dim=-2)),
        "hint": "The softmax is over the wrong axis: <code>dim=-1</code> normalises each row (one query's distribution over keys); <code>dim=-2</code> normalises each column. With the mask applied, several columns are entirely −∞, so this can also produce NaN.",
    },
    {
        "when": lambda fn: True,
        "hint": "Per head, it is exactly lesson 02: split Q and K, then <code>scores = Qh @ Kh.transpose(-2, -1) / sqrt(dh)</code>, mask with the causal triangle, <code>torch.softmax(scores, dim=-1)</code>.",
    },
]

# --------------------------------------------------------------------- Part D

PART_D_TESTS = [
    {
        "name": "returns (T × C)",
        "run": lambda fn: _shape_is(fn(randn(5, 6, seed=41), randn(5, 6, seed=42), randn(5, 6, seed=43), 2), (5, 6)),
    },
    {
        "name": "matches an independent implementation on random input",
        "run": lambda fn: match_detail(fn(P_Q, P_K, P_V, P_H), reference_multi_head(P_Q, P_K, P_V, P_H)),
    },
    {
        "name": "agrees with one fused kernel call per head",
        "run": lambda fn: _agrees_with_fused(fn),
    },
    {
        "name": "the first token's output is exactly its own value row",
        "run": lambda fn: _first_row_is_its_value(fn),
    },
    {
        "name": "a later value row cannot move an earlier output row",
        "run": lambda fn: _no_leak(fn),
    },
    {
        "name": "head h's output columns ignore head h′'s values",
        "run": lambda fn: _heads_do_not_mix(fn),
    },
    {
        "name": "one head is lesson 02's attention on the full width",
        "run": lambda fn: _one_head_matches_plain(fn),
    },
]

PART_D_DIAGNOSE = [
    {
        "when": lambda fn: near(fn(P_Q, P_K, P_V, P_H), _unmasked_reference(P_Q, P_K, P_V, P_H)),
        "hint": "Unmasked again: every head is averaging over the whole sequence, future included. <code>head_weights</code> is where the mask lives — if Part C passes, the mask is being lost between the two.",
    },
    {
        "when": lambda fn: _shape_is_raw(fn, (P_H, P_T, P_DH), (P_Q, P_K, P_V, P_H)),
        "hint": "You returned the heads separately: the output is (H, T, dh), not (T, C). After mixing the values, put the heads back side by side with <code>merge_heads</code>.",
    },
    {
        "when": lambda fn: near(fn(P_Q, P_K, P_V, P_H), reference_multi_head_reversed_heads(P_Q, P_K, P_V, P_H)),
        "hint": "The heads are concatenated in reverse: head 0's numbers are sitting in the last dh columns. Merge in the same order you split.",
    },
    {
        "when": lambda fn: near(fn(P_Q, P_K, P_V, P_H), reference_multi_head_upper(P_Q, P_K, P_V, P_H)),
        "hint": "The mask is upside down in every head: each token is reading only the future. <code>torch.triu</code> is the transpose of the causal mask — use <code>torch.tril</code>, with the diagonal visible.",
    },
    {
        "when": lambda fn: _shape_is_raw(fn, (P_T, P_H, P_DH), (P_Q, P_K, P_V, P_H)),
        "hint": "You merged (or split) in the wrong order — the result is (T, H, dh). <code>merge_heads</code> takes (H, T, dh) and returns (T, H·dh).",
    },
    {
        "when": lambda fn: True,
        "hint": "Three lines: <code>weights = head_weights(Q, K, n_head)</code>, <code>mixed = weights @ split_heads(V, n_head)</code>, <code>return merge_heads(mixed)</code>. If Part C is still failing, fix that first — this part can only be as right as the weights are.",
    },
]

PARTS = [
    {
        "id": "split",
        "title": "Part A — split_heads(x, n_head)",
        "entry": "split_heads",
        "missing_hint": "The checks look for a function named <code>split_heads</code> taking the tensor and the number of heads.",
        "tests": PART_A_TESTS,
        "diagnose": PART_A_DIAGNOSE,
    },
    {
        "id": "merge",
        "title": "Part B — merge_heads(x)",
        "entry": "merge_heads",
        "missing_hint": "The checks look for a function named <code>merge_heads</code> taking an (H, T, dh) tensor.",
        "tests": PART_B_TESTS,
        "diagnose": PART_B_DIAGNOSE,
    },
    {
        "id": "weights",
        "title": "Part C — head_weights(Q, K, n_head)",
        "entry": "head_weights",
        "missing_hint": "The checks look for a function named <code>head_weights</code> taking Q, K and the number of heads.",
        "tests": PART_C_TESTS,
        "diagnose": PART_C_DIAGNOSE,
    },
    {
        "id": "attention",
        "title": "Part D — multi_head_attention(Q, K, V, n_head)",
        "entry": "multi_head_attention",
        "missing_hint": "The checks look for a function named <code>multi_head_attention</code> taking Q, K, V and the number of heads.",
        "tests": PART_D_TESTS,
        "diagnose": PART_D_DIAGNOSE,
    },
]


# ------------------------------------------------------------------- internals


def _shape_is(value, wanted: tuple[int, ...]) -> tuple[bool, str]:
    tensor = as_tensor(value)
    text = "(" + " × ".join(str(d) for d in wanted) + ")"
    if tensor is None:
        return False, f"got {type(value).__name__}, wanted a tensor of shape {text}"
    if tuple(tensor.shape) != wanted:
        return False, f"got {shape_of(tensor)}, wanted {text}"
    return True, shape_of(tensor)


def _shape_is_raw(fn, wanted: tuple[int, ...], args) -> bool:
    """A diagnosis predicate: True when `fn(*args)` returns a tensor of `wanted` shape.

    Every part has a different call signature, so the arguments are passed in
    rather than guessed."""
    try:
        out = as_tensor(fn(*args))
    except BaseException:  # noqa: BLE001 - a wrong arity is simply not this diagnosis
        return False
    return out is not None and tuple(out.shape) == wanted


def _per_head_slice(fn) -> tuple[bool, str]:
    x = P_X
    heads = as_tensor(fn(x, P_H))
    if heads is None or tuple(heads.shape) != (P_H, P_T, P_DH):
        return False, f"got {shape_of(heads) if heads is not None else type(heads).__name__}, wanted ({P_H} × {P_T} × {P_DH})"
    for h in range(P_H):
        if not near(heads[h], x[:, h * P_DH:(h + 1) * P_DH]):
            return False, f"head {h} is not columns {h * P_DH}–{(h + 1) * P_DH - 1}: {shape_of(heads[h])}"
    return True, f"heads 0…{P_H - 1} are the width's contiguous slices, in order"


def _round_trip(fn) -> tuple[bool, str]:
    x = P_X
    split = reference_split(x, P_H)
    merged = as_tensor(fn(split))
    if merged is None or tuple(merged.shape) != (P_T, P_C):
        return False, f"got {shape_of(merged) if merged is not None else type(merged).__name__}, wanted ({P_T} × {P_C})"
    ok = torch.equal(merged, x)
    return ok, ("merge_heads(split_heads(x)) is x, bit for bit" if ok
                else f"the round trip changed the values: largest difference {(merged - x).abs().max().item():.3e}")


def _ordered_columns(fn) -> tuple[bool, str]:
    merged = as_tensor(fn(P_HEADS))
    if merged is None or tuple(merged.shape) != (P_T, P_C):
        return False, f"got {shape_of(merged) if merged is not None else type(merged).__name__}, wanted ({P_T} × {P_C})"
    for h in range(P_H):
        if not near(merged[:, h * P_DH:(h + 1) * P_DH], P_HEADS[h]):
            return False, f"columns {h * P_DH}–{(h + 1) * P_DH - 1} do not hold head {h}"
    return True, "column h·dh of the result is column 0 of head h"


def _raises_view_error(fn) -> bool:
    try:
        fn(P_HEADS)
    except RuntimeError as exc:
        return "view size is not compatible" in str(exc)
    except BaseException:  # noqa: BLE001
        return False
    return False


def _rows_sum_to_one(fn) -> tuple[bool, str]:
    weights = as_tensor(fn(P_Q, P_K, P_H))
    if weights is None or tuple(weights.shape) != (P_H, P_T, P_T):
        return False, f"got {shape_of(weights) if weights is not None else type(weights).__name__}, wanted ({P_H} × {P_T} × {P_T})"
    sums = weights.detach().sum(dim=-1)
    worst = (sums - 1).abs().max().item()
    return worst <= ATOL, f"worst row sum across all heads is off by {worst:.2e}"


def _blocked_are_zero(fn) -> tuple[bool, str]:
    weights = as_tensor(fn(P_Q, P_K, P_H))
    if weights is None or tuple(weights.shape) != (P_H, P_T, P_T):
        return False, f"got {shape_of(weights) if weights is not None else type(weights).__name__}, wanted ({P_H} × {P_T} × {P_T})"
    mask = torch.tril(torch.ones(P_T, P_T))
    blocked = weights.detach()[:, mask == 0]
    nonzero = int((blocked != 0).sum())
    total = blocked.numel()
    if nonzero == 0:
        return True, f"all {total} blocked cells across {P_H} heads are exactly 0.0"
    return False, f"{nonzero} of {total} blocked cells hold weight (largest {blocked.abs().max().item():.4f})"


def _heads_read_their_own_slice(fn) -> tuple[bool, str]:
    K2 = P_K.clone()
    K2[:, P_DH:] += 5.0                     # disturb head 1's keys only
    before = as_tensor(fn(P_Q, P_K, P_H))
    after = as_tensor(fn(P_Q, K2, P_H))
    if before is None or tuple(before.shape) != (P_H, P_T, P_T):
        return False, f"got {shape_of(before) if before is not None else type(before).__name__}, wanted ({P_H} × {P_T} × {P_T})"
    kept = bool(torch.equal(before.detach()[0], after.detach()[0]))
    moved = not bool(torch.equal(before.detach()[1], after.detach()[1]))
    return kept and moved, ("head 0's weights are bit-identical after head 1's keys changed; head 1's did change"
                            if kept and moved else
                            "a change in head 1's keys moved head 0's weights — the slices are crossing over")


def _one_head_is_plain(fn) -> tuple[bool, str]:
    out = as_tensor(fn(P_Q, P_K, 1))
    if out is None or tuple(out.shape) != (1, P_T, P_T):
        return False, f"got {shape_of(out) if out is not None else type(out).__name__}, wanted (1 × {P_T} × {P_T})"
    scores = P_Q @ P_K.transpose(-2, -1) / math.sqrt(P_C)
    causal = torch.tril(torch.ones(P_T, P_T))
    plain = torch.softmax(scores.masked_fill(causal == 0, float("-inf")), dim=-1)
    diff = (out.detach()[0].double() - plain.double()).abs().max().item()
    return diff <= ATOL, f"one head of width {P_C} reproduces lesson 02's weights (largest difference {diff:.1e})"


def _unmasked_reference(Q, K, V, n_head):
    Qh, Kh, Vh = reference_split(Q, n_head), reference_split(K, n_head), reference_split(V, n_head)
    weights = torch.softmax(Qh @ Kh.transpose(-2, -1) / math.sqrt(Qh.shape[-1]), dim=-1)
    return reference_merge(weights @ Vh)


def _agrees_with_fused(fn) -> tuple[bool, str]:
    mine = as_tensor(fn(P_Q, P_K, P_V, P_H))
    fused = fused_multi_head(P_Q, P_K, P_V, P_H)
    if mine is None or tuple(mine.shape) != tuple(fused.shape):
        return False, f"got {shape_of(mine) if mine is not None else type(mine).__name__}, wanted {shape_of(fused)}"
    diff = (mine.detach().double() - fused.double()).abs().max().item()
    return diff <= ATOL, f"your per-head lines and the fused kernel agree to {diff:.1e}"


def _first_row_is_its_value(fn) -> tuple[bool, str]:
    Q, K, V = randn(4, 8, seed=51), randn(4, 8, seed=52), randn(4, 8, seed=53)
    out = as_tensor(fn(Q, K, V, 4))
    if out is None or tuple(out.shape) != (4, 8):
        return False, f"got {shape_of(out) if out is not None else type(out).__name__}, wanted (4 × 8) before this check can run"
    first = out.detach()[0]
    ok = bool((first - V[0]).abs().max().item() <= ATOL)
    return ok, ("row 0 is V[0] across every head — with only itself visible there is nothing to average" if ok
                else f"row 0 is {[round(x, 3) for x in first.tolist()[:4]]}… but V[0] starts {[round(x, 3) for x in V[0].tolist()[:4]]}")


def _no_leak(fn) -> tuple[bool, str]:
    Q, K, V = randn(6, 8, seed=61), randn(6, 8, seed=62), randn(6, 8, seed=63)
    base = as_tensor(fn(Q, K, V, 4))
    if base is None or tuple(base.shape) != (6, 8):
        return False, f"got {shape_of(base) if base is not None else type(base).__name__}, wanted (6 × 8) before this check can run"
    tampered = V.clone()
    tampered[4:] = tampered[4:] + 500.0
    after = as_tensor(fn(Q, K, tampered, 4))
    if after is None or tuple(after.shape) != (6, 8):
        return False, f"got {shape_of(after) if after is not None else type(after).__name__} on the tampered values"
    kept = torch.equal(base.detach()[:4], after.detach()[:4])
    moved = not torch.equal(base.detach()[4:], after.detach()[4:])
    return kept and moved, ("rows 0–3 are bit-identical after value rows 4–5 were replaced; row 4 did change"
                            if kept and moved else
                            "tampering with value rows 4–5 moved output rows 0–3 — the future is leaking backwards")


def _heads_do_not_mix(fn) -> tuple[bool, str]:
    n_head, T, C = 4, 5, 8
    dh = C // n_head
    Q, K, V = randn(T, C, seed=71), randn(T, C, seed=72), randn(T, C, seed=73)
    base = as_tensor(fn(Q, K, V, n_head))
    if base is None or tuple(base.shape) != (T, C):
        return False, f"got {shape_of(base) if base is not None else type(base).__name__}, wanted ({T} × {C})"
    tampered = V.clone()
    tampered[:, dh:2 * dh] += 99.0          # head 1's slice of the values
    after = as_tensor(fn(Q, K, tampered, n_head))
    if after is None or tuple(after.shape) != (T, C):
        return False, f"got {shape_of(after) if after is not None else type(after).__name__} on the tampered values"
    kept = bool(torch.equal(base.detach()[:, :dh], after.detach()[:, :dh])) and bool(
        torch.equal(base.detach()[:, 2 * dh:], after.detach()[:, 2 * dh:]))
    moved = not bool(torch.equal(base.detach()[:, dh:2 * dh], after.detach()[:, dh:2 * dh]))
    return kept and moved, (f"the other {n_head - 1} heads' columns are bit-identical; head 1's moved"
                            if kept and moved else
                            "changing one head's values moved another head's output columns — the concatenation is mixing heads")


def _one_head_matches_plain(fn) -> tuple[bool, str]:
    Q, K, V = randn(6, 8, seed=81), randn(6, 8, seed=82), randn(6, 8, seed=83)
    out = as_tensor(fn(Q, K, V, 1))
    if out is None or tuple(out.shape) != (6, 8):
        return False, f"got {shape_of(out) if out is not None else type(out).__name__}, wanted (6 × 8)"
    plain = plain_causal_attention(Q, K, V)
    diff = (out.detach().double() - plain.double()).abs().max().item()
    return diff <= ATOL, f"with n_head = 1 your whole block collapses to lesson 02's causal attention (largest difference {diff:.1e})"


# --------------------------------------------------------------------- figures

TOKENS = ["the", "trophy", "fit", "it"]
N_HEAD = 2
Q = torch.tensor([[0.2, 0.9, 0.1, 0.4],
                  [0.9, 0.2, 0.5, 0.1],
                  [0.6, 0.4, 0.7, 0.2],
                  [1.0, 0.0, 0.3, 0.8]])
K = torch.tensor([[0.2, 0.9, 0.3, 0.2],
                  [1.0, 0.1, 0.4, 0.6],
                  [0.6, 0.5, 0.2, 0.7],
                  [0.3, 0.7, 0.9, 0.1]])
V = torch.tensor([[0.1, 0.1, 0.1, 0.9],
                  [1.0, 0.2, 0.0, 0.3],
                  [0.0, 0.6, 0.4, 0.2],
                  [0.2, 0.2, 0.6, 0.5]])


def _entry(namespace, name):
    fn = namespace.get(name)
    return fn if callable(fn) else None


def _run(fn, args):
    if fn is None:
        return {"source": "absent", "values": None, "error": None}
    try:
        out = as_tensor(fn(*args))
    except BaseException as exc:  # noqa: BLE001
        return {"source": "error", "values": None, "error": f"{type(exc).__name__}: {exc}"}
    if out is None:
        return {"source": "error", "values": None, "error": "did not return numbers"}
    return {"source": "learner", "values": out.detach().tolist()}


def figures(namespace) -> dict:
    """Numbers for the lesson page: whatever the learner's code produces right now."""
    T = len(TOKENS)
    return {
        "tokens": TOKENS,
        "n_head": N_HEAD,
        "d_head": int(Q.shape[-1] // N_HEAD),
        "width": int(Q.shape[-1]),
        "mask_used": {"source": "reference", "values": torch.tril(torch.ones(T, T)).int().tolist()},
        "split": _run(_entry(namespace, "split_heads"), (Q, N_HEAD)),
        "heads": _run(_entry(namespace, "head_weights"), (Q, K, N_HEAD)),
        "output": _run(_entry(namespace, "multi_head_attention"), (Q, K, V, N_HEAD)),
    }
