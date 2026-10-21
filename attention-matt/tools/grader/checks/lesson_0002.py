"""lesson_0002.py — checks for "The mask and the scale".

The references below are written independently of `exercises/`. Causal masking
is the single edit that makes a decoder-only model possible, so the three parts
test it in three separable pieces: the mask itself, the masked weights, and the
end-to-end property that no token can influence an earlier one.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .support import ATOL, as_tensor, match_detail, near, randn, scaled_scores, shape_of

# ---------------------------------------------------------------- references


def reference_mask(T: int) -> torch.Tensor:
    """The causal triangle: 1 on and below the diagonal, 0 above it."""
    return torch.tril(torch.ones(T, T))


def reference_masked_weights(Q: torch.Tensor, K: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    scores = Q @ K.transpose(-2, -1) / math.sqrt(K.shape[-1])
    blocked = as_tensor(mask) == 0
    return torch.softmax(scores.masked_fill(blocked, float("-inf")), dim=-1)


def reference_causal_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    return reference_masked_weights(Q, K, reference_mask(Q.shape[-2])) @ V


def plain_weights(Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Lesson 01's answer: no mask anywhere."""
    return torch.softmax(scaled_scores(Q, K), dim=-1)


def scattered_mask(T: int, seed: int) -> torch.Tensor:
    """A 0/1 mask with the diagonal forced visible, so no row is all-blocked."""
    keep = randn(T, T, seed=seed) > 0
    return (keep | torch.eye(T, dtype=torch.bool)).float()


# Fixed probe inputs for the diagnoses: square, so a wrong-but-plausible
# implementation still *returns* a tensor and can be named instead of raising.
P_T, P_D = 4, 4
P_Q = randn(P_T, P_D, seed=101)
P_K = randn(P_T, P_D, seed=102)
P_V = randn(P_T, P_D, seed=103)
P_MASK = reference_mask(P_T)
P_SCATTER = scattered_mask(P_T, seed=104)
P_UPPER = torch.triu(torch.ones(P_T, P_T))


def _polarity_probe() -> torch.Tensor:
    """A 0/1 mask where neither it nor its inverse fully blocks a row.

    The causal mask has an all-visible last row, and inverting that row would
    make the softmax return NaN, which cannot be compared against anything —
    so the polarity diagnosis gets its own probe."""
    keep = (randn(P_T, P_T, seed=105) > 0) | torch.eye(P_T, dtype=torch.bool)
    keep[:, -1] = False  # every row keeps at least one blocked cell in both directions
    return keep.float()


P_POLAR = _polarity_probe()


# --------------------------------------------------------------------- Part A

PART_A_TESTS = [
    {
        "name": "returns a (T × T) tensor of 0s and 1s",
        "run": lambda fn: _binary_square(fn(5)),
    },
    {
        "name": "position i may look at keys 0..i, and no further",
        "run": lambda fn: match_detail(fn(6), reference_mask(6), atol=0.0),
    },
    {
        "name": "a token is allowed to look at itself",
        "run": lambda fn: _diagonal_visible(fn(4)),
    },
    {
        "name": "a one-token sequence has one visible cell",
        "run": lambda fn: match_detail(fn(1), reference_mask(1), atol=0.0),
    },
]

PART_A_DIAGNOSE = [
    {
        "when": lambda fn: near(fn(P_T), torch.tril(torch.ones(P_T, P_T), diagonal=-1)),
        "hint": "You built the strictly lower triangle, so the diagonal is 0 and every token blocks <em>itself</em>. A token has to be allowed to look at its own position — include the diagonal (<code>torch.tril(torch.ones(T, T))</code>, no <code>diagonal=-1</code>).",
    },
    {
        "when": lambda fn: near(fn(P_T), P_UPPER),
        "hint": "That is the <em>upper</em> triangle: you allowed every token to look only at what comes after it and blocked the past. <code>torch.triu</code> is the transpose of what a causal mask needs — use <code>torch.tril</code>.",
    },
    {
        "when": lambda fn: near(fn(P_T), torch.ones(P_T, P_T)),
        "hint": "Every cell is visible, so this mask blocks nothing at all — it is the lesson-01 behaviour with extra steps. Rows must stop after column <em>i</em>.",
    },
    {
        "when": lambda fn: near(fn(P_T), torch.zeros(P_T, P_T)),
        "hint": "Nothing is visible, so every score row becomes all −∞ and the softmax has nothing to distribute mass over. Row <em>i</em> has to keep at least its own key.",
    },
    {
        "when": lambda fn: True,
        "hint": "One line: <code>torch.tril(torch.ones(T, T))</code> — 1 where a query may look, 0 where it may not. Check it with <code>causal_mask(4)</code> in a REPL: the first row should be <code>[1, 0, 0, 0]</code>.",
    },
]

# --------------------------------------------------------------------- Part B

PART_B_TESTS = [
    {
        "name": "returns a (T × T) weight matrix",
        "run": lambda fn: _shape_test(fn(randn(5, 4, seed=11), randn(5, 4, seed=12), scattered_mask(5, seed=13)), (5, 5)),
    },
    {
        "name": "matches the reference on a scattered mask",
        "run": lambda fn: match_detail(fn(P_Q, P_K, P_SCATTER), reference_masked_weights(P_Q, P_K, P_SCATTER)),
    },
    {
        "name": "matches the reference on the causal mask",
        "run": lambda fn: match_detail(fn(P_Q, P_K, P_MASK), reference_masked_weights(P_Q, P_K, P_MASK)),
    },
    {
        "name": "blocked cells are exactly 0, not nearly 0",
        "run": lambda fn: _blocked_are_zero(fn),
    },
    {
        "name": "every row still sums to 1",
        "run": lambda fn: _rows_sum_to_one(fn),
    },
    {
        "name": "an all-visible mask gives back plain attention",
        "run": lambda fn: match_detail(fn(P_Q, P_K, torch.ones(P_T, P_T)), plain_weights(P_Q, P_K)),
    },
    {
        "name": "leaving one key per row visible gives one-hot rows",
        "run": lambda fn: match_detail(fn(P_Q, P_K, torch.eye(P_T)), torch.eye(P_T)),
    },
]

PART_B_DIAGNOSE = [
    {
        "when": lambda fn: near(fn(P_Q, P_K, P_MASK), plain_weights(P_Q, P_K)),
        "hint": "This is unmasked attention — every token is still looking at the whole sequence. The mask argument is being ignored; the scores have to be edited with it before the softmax.",
    },
    {
        "when": lambda fn: near(fn(P_Q, P_K, P_MASK), torch.softmax(scaled_scores(P_Q, P_K).masked_fill(P_MASK == 0, 0.0), dim=-1)),
        "hint": "You set the blocked scores to <code>0</code> instead of −∞. <code>exp(0) = 1</code>, so every blocked key comes back with a real share of the row — the mask has to make the score so negative that <code>exp</code> underflows to exactly 0.",
    },
    {
        "when": lambda fn: near(fn(P_Q, P_K, P_MASK), plain_weights(P_Q, P_K).masked_fill(P_MASK == 0, 0.0)),
        "hint": "You masked <em>after</em> the softmax. Zeroing blocked weights leaves the row summing to less than 1, so it is no longer a distribution: normalisation must happen over the visible keys only, which means masking first.",
    },
    {
        "when": lambda fn: near(fn(P_Q, P_K, P_POLAR), reference_masked_weights(P_Q, P_K, 1.0 - P_POLAR)),
        "hint": "You kept the wrong side of the mask: the output equals the reference computed with the mask inverted. <code>1</code> means <em>may look</em>, so <code>0</code> is the cell to block.",
    },
    {
        "when": lambda fn: near(fn(P_Q, P_K, P_MASK), torch.softmax((P_Q @ P_K.transpose(-2, -1)).masked_fill(P_MASK == 0, float("-inf")), dim=-1)),
        "hint": "The mask is in the right place, but the scores were never divided by <code>√d<sub>k</sub></code> — that step does not disappear when a mask arrives.",
    },
    {
        "when": lambda fn: near(fn(P_Q, P_K, P_MASK), scaled_scores(P_Q, P_K).masked_fill(P_MASK == 0, float("-inf"))),
        "hint": "You scaled and masked the scores but never ran the softmax, so these are still similarities — one of them is −∞. <code>torch.softmax(scores, dim=-1)</code> is what turns them into weights.",
    },
    {
        "when": lambda fn: near(fn(P_Q, P_K, P_MASK), torch.softmax(scaled_scores(P_Q, P_K).masked_fill(P_MASK == 0, float("-inf")), dim=-2)),
        "hint": "The <code>dim</code> is wrong: you normalised down the columns (<code>dim=-2</code>) instead of along each row. One row is one query's distribution — <code>dim=-1</code>, as in lesson 01.",
    },
    {
        "when": lambda fn: True,
        "hint": "Order: <code>scores = Q @ K.transpose(-2, -1)</code>, then <code>/ dk ** 0.5</code>, then <code>scores.masked_fill(mask == 0, float('-inf'))</code>, then <code>torch.softmax(scores, dim=-1)</code>. Only the last two have to be in that order.",
    },
]

# --------------------------------------------------------------------- Part C

PART_C_TESTS = [
    {
        "name": "returns a (T × dv) output",
        "run": lambda fn: _shape_test(fn(randn(5, 4, seed=21), randn(5, 4, seed=22), randn(5, 7, seed=23)), (5, 7)),
    },
    {
        "name": "matches the reference on random input",
        "run": lambda fn: match_detail(
            fn(randn(6, 5, seed=31), randn(6, 5, seed=32), randn(6, 9, seed=33)),
            reference_causal_attention(randn(6, 5, seed=31), randn(6, 5, seed=32), randn(6, 9, seed=33)),
        ),
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
        "name": "agrees with F.scaled_dot_product_attention(is_causal=True)",
        "run": lambda fn: _agrees_with_fused(fn),
    },
    {
        "name": "a one-token sequence just returns its value",
        "run": lambda fn: _single_token(fn),
    },
]

PART_C_DIAGNOSE = [
    {
        "when": lambda fn: near(fn(P_Q, P_K, P_V), plain_weights(P_Q, P_K) @ P_V),
        "hint": "This is unmasked attention again — every token is mixing in the future. <code>causal_attention</code> has to build its own mask; nothing upstream hands it one.",
    },
    {
        "when": lambda fn: near(fn(P_Q, P_K, P_V), reference_masked_weights(P_Q, P_K, P_UPPER) @ P_V),
        "hint": "The mask is upside down: the output equals causal attention on a <em>reversed</em> sequence, i.e. each token saw only the future. Block the cells with <em>j &gt; i</em>, not <em>j &lt; i</em>.",
    },
    {
        "when": lambda fn: near(fn(P_Q, P_K, P_V), (plain_weights(P_Q, P_K).masked_fill(P_MASK == 0, 0.0)) @ P_V),
        "hint": "The mask arrived after the softmax, so the blocked positions were removed from an already-normalised row and the row no longer sums to 1. Mask the scores, then normalise.",
    },
    {
        "when": lambda fn: near(fn(P_Q, P_K, P_V), torch.softmax(scaled_scores(P_Q, P_K).masked_fill(P_MASK == 0, 0.0), dim=-1) @ P_V),
        "hint": "The blocked scores were set to <code>0</code> rather than −∞, so they came back with <code>exp(0) = 1</code> of influence over the output. The future is leaking into every row.",
    },
    {
        "when": lambda fn: near(fn(P_Q, P_K, P_V), reference_masked_weights(P_Q, P_K, P_MASK)),
        "hint": "You returned the weights, not the output. The masked weights are (T × T); the answer is those weights multiplied by <code>V</code>.",
    },
    {
        "when": lambda fn: True,
        "hint": "Two lines: <code>mask = causal_mask(Q.shape[-2])</code> then <code>masked_weights(Q, K, mask) @ V</code>. If Parts A and B are still failing, they explain everything here — fix them first.",
    },
]

PARTS = [
    {
        "id": "mask",
        "title": "Part A — causal_mask(T)",
        "entry": "causal_mask",
        "missing_hint": "The checks look for a function named <code>causal_mask</code> that takes the sequence length <code>T</code>.",
        "tests": PART_A_TESTS,
        "diagnose": PART_A_DIAGNOSE,
    },
    {
        "id": "weights",
        "title": "Part B — masked_weights(Q, K, mask)",
        "entry": "masked_weights",
        "missing_hint": "The checks look for a function named <code>masked_weights</code> taking three arguments (Q, K, mask).",
        "tests": PART_B_TESTS,
        "diagnose": PART_B_DIAGNOSE,
    },
    {
        "id": "causal",
        "title": "Part C — causal_attention(Q, K, V)",
        "entry": "causal_attention",
        "missing_hint": "The checks look for a function named <code>causal_attention</code> taking three arguments (Q, K, V).",
        "tests": PART_C_TESTS,
        "diagnose": PART_C_DIAGNOSE,
    },
]


# ------------------------------------------------------------------- internals


def _shape_test(result, wanted: tuple[int, int]) -> tuple[bool, str]:
    wanted_text = f"({wanted[0]} × {wanted[1]})"
    tensor = as_tensor(result)
    if tensor is None:
        return False, f"got {type(result).__name__}, wanted a tensor of shape {wanted_text}"
    if tensor.dim() != 2:
        return False, f"got {shape_of(tensor)}, wanted {wanted_text} — these functions work on plain 2-D (T, d) tensors"
    if tuple(tensor.shape) != wanted:
        return False, f"got {shape_of(tensor)}, wanted {wanted_text}"
    return True, f"{shape_of(tensor)}"


def _binary_square(value) -> tuple[bool, str]:
    tensor = as_tensor(value)
    if tensor is None:
        return False, f"got {type(value).__name__}, wanted a 2-D tensor of shape (T × T)"
    if tensor.dim() != 2 or tensor.shape[0] != tensor.shape[1]:
        return False, f"got {shape_of(tensor)}, wanted a square (T × T) mask"
    values = torch.unique(tensor.detach().to(torch.float32))
    if values.numel() > 2 or bool((((values == 0) | (values == 1)).logical_not()).any()):
        return False, f"cells hold {values.tolist()[:6]} — a mask is 0 or 1 per cell"
    return True, f"{shape_of(tensor)}, values {values.tolist()}"


def _diagonal_visible(value, T: int = 4) -> tuple[bool, str]:
    tensor = as_tensor(value)
    if tensor is None or tuple(tensor.shape) != (T, T):
        return False, f"got {shape_of(tensor) if tensor is not None else type(value).__name__}, wanted ({T} × {T})"
    diagonal = torch.diagonal(tensor.detach().to(torch.float32))
    ok = bool((diagonal == 1).all())
    return ok, (f"the diagonal is 1 for all {T} tokens" if ok
                else f"the diagonal is {diagonal.tolist()} — a token must be able to see its own position")


def _blocked_are_zero(fn) -> tuple[bool, str]:
    weights = as_tensor(fn(P_Q, P_K, P_SCATTER))
    if weights is None or tuple(weights.shape) != (P_T, P_T):
        return False, f"got {shape_of(weights) if weights is not None else type(weights).__name__}, wanted (4 × 4)"
    blocked = P_SCATTER == 0
    values = weights.detach()[blocked]
    nonzero = int((values != 0).sum())
    count = int(blocked.sum())
    if nonzero == 0:
        return True, f"all {count} blocked cells are exactly 0.0, so they contribute nothing to the output"
    return False, (f"{nonzero} of {count} blocked cells still hold weight (largest {values.abs().max().item():.4f}) — "
                   f"a finite filler like 0 leaves exp(0) = 1 of mass in the row")


def _rows_sum_to_one(fn) -> tuple[bool, str]:
    weights = as_tensor(fn(P_Q, P_K, P_SCATTER))
    if weights is None or weights.dim() != 2:
        return False, f"got {shape_of(weights) if weights is not None else type(weights).__name__}, cannot sum its rows"
    sums = weights.detach().sum(dim=-1)
    worst = (sums - 1).abs().max().item()
    row = int((sums - 1).abs().argmax().item())
    return worst <= ATOL, f"row {row} sums to {sums[row].item():.6f}, wanted 1 (worst error {worst:.2e})"


def _first_row_is_its_value(fn) -> tuple[bool, str]:
    Q, K, V = randn(4, 3, seed=41), randn(4, 3, seed=42), randn(4, 5, seed=43)
    out = as_tensor(fn(Q, K, V))
    if out is None or tuple(out.shape) != (4, 5):
        return False, f"got {shape_of(out) if out is not None else type(out).__name__}, wanted (4 × 5) before this check can run"
    first, value = out.detach()[0], V[0]
    ok = torch.equal(first, value) or bool((first - value).abs().max().item() <= ATOL)
    return ok, (f"row 0 is V[0] exactly — with only itself visible there is nothing to average" if ok
                else f"row 0 is {[round(x, 4) for x in first.tolist()]} but V[0] is {[round(x, 4) for x in value.tolist()]}")


def _no_leak(fn) -> tuple[bool, str]:
    Q, K, V = randn(6, 4, seed=51), randn(6, 4, seed=52), randn(6, 7, seed=53)
    base = as_tensor(fn(Q, K, V))
    if base is None or tuple(base.shape) != (6, 7):
        return False, f"got {shape_of(base) if base is not None else type(base).__name__}, wanted (6 × 7) before this check can run"
    tampered = V.clone()
    tampered[4:] = tampered[4:] + 500.0
    after = as_tensor(fn(Q, K, tampered))
    if after is None or tuple(after.shape) != (6, 7):
        return False, f"got {shape_of(after) if after is not None else type(after).__name__} on the tampered values"
    kept = torch.equal(base.detach()[:4], after.detach()[:4])
    moved = not torch.equal(base.detach()[4:], after.detach()[4:])
    return kept and moved, (f"rows 0–3 are bit-identical after value rows 4–5 were replaced with something absurd; row 4 did change"
                            if kept and moved else
                            f"tampering with value rows 4–5 moved output rows 0–3 — the future is leaking backwards")


def _agrees_with_fused(fn) -> tuple[bool, str]:
    Q, K, V = randn(6, 5, seed=61), randn(6, 5, seed=62), randn(6, 8, seed=63)
    mine = as_tensor(fn(Q, K, V))
    fused = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    if mine is None or tuple(mine.shape) != tuple(fused.shape):
        return False, f"got {shape_of(mine) if mine is not None else type(mine).__name__}, wanted {shape_of(fused)}"
    diff = (mine.detach().double() - fused.double()).abs().max().item()
    return diff <= ATOL, f"your five lines and the fused kernel agree to {diff:.1e}"


def _single_token(fn) -> tuple[bool, str]:
    Q, K, V = randn(1, 3, seed=71), randn(1, 3, seed=72), randn(1, 4, seed=73)
    out = as_tensor(fn(Q, K, V))
    if out is None or tuple(out.shape) != (1, 4):
        return False, f"got {shape_of(out) if out is not None else type(out).__name__}, wanted (1 × 4)"
    return bool((out.detach()[0] - V[0]).abs().max().item() <= ATOL), f"got {[round(x, 4) for x in out[0].tolist()]}, wanted V[0]"


# --------------------------------------------------------------------- figures

TOKENS = ["the", "trophy", "fit", "it"]
Q = torch.tensor([[0.2, 0.9], [0.9, 0.2], [0.6, 0.4], [1.0, 0.0]])
K = torch.tensor([[0.2, 0.9], [1.0, 0.1], [0.6, 0.5], [0.3, 0.7]])
V = torch.tensor([[0.1, 0.1, 0.1], [1.0, 0.2, 0.0], [0.0, 0.6, 0.4], [0.2, 0.2, 0.6]])


def _entry(namespace, name):
    fn = namespace.get(name)
    return fn if callable(fn) else None


def _run(fn, args, integer: bool = False):
    if fn is None:
        return {"source": "absent", "values": None, "error": None}
    try:
        out = as_tensor(fn(*args))
    except BaseException as exc:  # noqa: BLE001
        return {"source": "error", "values": None, "error": f"{type(exc).__name__}: {exc}"}
    if out is None:
        return {"source": "error", "values": None, "error": "did not return numbers"}
    values = out.detach().tolist()
    if integer:
        values = [[int(x) for x in row] for row in values]
    return {"source": "learner", "values": values}


def figures(namespace) -> dict:
    """Numbers for the lesson page: whatever the learner's code produces right now."""
    T = len(TOKENS)
    return {
        "tokens": TOKENS,
        "dk": int(Q.shape[-1]),
        # the weights are drawn with the *reference* triangle, so a broken mask
        # in Part A cannot hide what Part B already gets right
        "mask_used": {"source": "reference", "values": reference_mask(T).int().tolist()},
        "mask": _run(_entry(namespace, "causal_mask"), (T,), integer=True),
        "weights": _run(_entry(namespace, "masked_weights"), (Q, K, reference_mask(T))),
        "output": _run(_entry(namespace, "causal_attention"), (Q, K, V)),
    }
