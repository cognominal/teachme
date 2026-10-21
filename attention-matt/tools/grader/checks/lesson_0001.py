"""lesson_0001.py — checks for "Attention is a weighted average".

The reference implementations below are written independently of anything in
`exercises/`: they are the target the learner's code is compared against, and
the source of every named diagnosis.
"""

from __future__ import annotations

import math

import torch

from .support import ATOL, as_tensor, describe, match_detail, near, randn, scaled_scores, shape_of

# ---------------------------------------------------------------- references


def reference_weights(Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    scores = Q @ K.transpose(-2, -1) / math.sqrt(K.shape[-1])
    return torch.softmax(scores, dim=-1)


def reference_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    return reference_weights(Q, K) @ V


# --------------------------------------------------------------------- Part A

PART_A_TESTS = [
    {
        "name": "returns a (T × T) matrix",
        "run": lambda fn: _shape_test(fn(randn(5, 4, seed=11), randn(5, 4, seed=12)), (5, 5)),
    },
    {
        "name": "every row sums to 1",
        "run": lambda fn: _rows_sum_to_one(fn),
    },
    {
        "name": "matches the reference on random input",
        "run": lambda fn: match_detail(fn(randn(4, 6, seed=21), randn(4, 6, seed=22)), reference_weights(randn(4, 6, seed=21), randn(4, 6, seed=22))),
    },
    {
        "name": "a one-token sequence attends entirely to itself",
        "run": lambda fn: _single_token(fn(randn(1, 3, seed=31), randn(1, 3, seed=32))),
    },
]

PART_A_DIAGNOSE = [
    {
        "when": lambda fn: near(fn(randn(4, 6, seed=41), randn(4, 6, seed=42)), torch.softmax(randn(4, 6, seed=41) @ randn(4, 6, seed=42).transpose(-2, -1), dim=-1)),
        "hint": "You skipped the scale: the output is exactly <code>softmax(Q @ K.transpose(-2, -1))</code>. With <code>d_k = 6</code> the raw scores are √6 ≈ 2.4× too large, so the distribution collapses onto one key.",
    },
    {
        "when": lambda fn: near(fn(randn(4, 6, seed=41), randn(4, 6, seed=42)), torch.softmax(randn(4, 6, seed=42) @ randn(4, 6, seed=41).transpose(-2, -1) / math.sqrt(6), dim=-1)),
        "hint": "You computed <code>K @ Q.transpose(-2, -1)</code>. The query has to be on the left: row <em>i</em> must hold query <em>i</em>'s pulls over all keys.",
    },
    {
        "when": lambda fn: near(fn(randn(4, 6, seed=41), randn(4, 6, seed=42)), torch.softmax(scaled_scores(randn(4, 6, seed=41), randn(4, 6, seed=42)), dim=-2)),
        "hint": "The scale is right, but you normalised down the columns (<code>dim=-2</code> or <code>dim=0</code>) instead of across the rows. One row is one query's distribution over the keys — <code>dim=-1</code>.",
    },
    {
        "when": lambda fn: near(fn(randn(4, 6, seed=41), randn(4, 6, seed=42)), scaled_scores(randn(4, 6, seed=41), randn(4, 6, seed=42))),
        "hint": "You scaled the scores but never applied the softmax, so the rows are raw similarities rather than weights. <code>torch.softmax(..., dim=-1)</code> is what makes each row non-negative and summing to 1.",
    },
    {
        "when": lambda fn: near(fn(randn(4, 6, seed=41), randn(4, 6, seed=42)), randn(4, 6, seed=41) @ randn(4, 6, seed=42).transpose(-2, -1)),
        "hint": "That is the raw score matrix — no scale and no softmax. Similarities become weights only after <code>torch.softmax</code>.",
    },
    {
        "when": lambda fn: True,
        "hint": "Build it in the order of the formula: <code>scores = Q @ K.transpose(-2, -1)</code>, then <code>scores / dk ** 0.5</code> with <code>dk = Q.shape[-1]</code>, then <code>torch.softmax(scores, dim=-1)</code>.",
    },
]

# --------------------------------------------------------------------- Part B

PART_B_TESTS = [
    {
        "name": "returns a (T × dv) matrix",
        "run": lambda fn: _shape_test(fn(randn(3, 4, seed=51), randn(3, 4, seed=52), randn(3, 7, seed=53)), (3, 7)),
    },
    {
        "name": "matches the reference on random input",
        "run": lambda fn: match_detail(
            fn(randn(5, 6, seed=61), randn(5, 6, seed=62), randn(5, 9, seed=63)),
            reference_attention(randn(5, 6, seed=61), randn(5, 6, seed=62), randn(5, 9, seed=63)),
        ),
    },
    {
        "name": "each output row stays inside the values it mixes",
        "run": lambda fn: _convex(fn),
    },
]

PART_B_DIAGNOSE = [
    {
        "when": lambda fn: near(fn(randn(4, 4, seed=71), randn(4, 4, seed=72), randn(4, 4, seed=73)), reference_weights(randn(4, 4, seed=71), randn(4, 4, seed=72))),
        "hint": "You returned the weights themselves, so the output is still (T × T). The weights say <em>how much</em> of each value row to take; they are not the answer until they multiply <code>V</code>.",
    },
    {
        "when": lambda fn: near(fn(randn(4, 4, seed=81), randn(4, 4, seed=82), randn(4, 4, seed=83)), reference_weights(randn(4, 4, seed=81), randn(4, 4, seed=82)) @ randn(4, 4, seed=83).transpose(-2, -1)),
        "hint": "You multiplied by <code>V.transpose(-2, -1)</code>. The weights are (T × T) and <code>V</code> is (T × dv); to mix <em>rows</em> of V, V goes on the right un-transposed.",
    },
    {
        "when": lambda fn: near(
            fn(randn(4, 6, seed=91), randn(4, 6, seed=92), randn(4, 5, seed=93)),
            torch.softmax(randn(4, 6, seed=91) @ randn(4, 6, seed=92).transpose(-2, -1), dim=-1) @ randn(4, 5, seed=93),
        ),
        "hint": "The mixing is right, but the weights here have no <code>1/√d_k</code> scale — your <code>attention_weights</code> was supposed to supply it. Go back and fix Part A.",
    },
    {
        "when": lambda fn: True,
        "hint": "Part B is one multiplication: <code>attention_weights(Q, K) @ V</code>. These numbers can only be as right as the weights are — if Part A is still failing, fix that first.",
    },
]

PARTS = [
    {
        "id": "weights",
        "title": "Part A — attention_weights(Q, K)",
        "entry": "attention_weights",
        "missing_hint": "The checks look for a function named <code>attention_weights</code> in <code>exercises/0001_attention.py</code>.",
        "tests": PART_A_TESTS,
        "diagnose": PART_A_DIAGNOSE,
    },
    {
        "id": "attention",
        "title": "Part B — attention(Q, K, V)",
        "entry": "attention",
        "missing_hint": "The checks look for a function named <code>attention</code> that takes three arguments (Q, K, V).",
        "tests": PART_B_TESTS,
        "diagnose": PART_B_DIAGNOSE,
    },
]


# ------------------------------------------------------------------- internals


def _shape_test(result, wanted: tuple[int, int]) -> tuple[bool, str]:
    wanted_text = f"({wanted[0]} × {wanted[1]})"
    tensor = as_tensor(result)
    if tensor is None:
        return False, f"got {type(result).__name__}, wanted a tensor of shape {wanted_text}"
    if tensor.dim() != 2:
        return False, f"got {shape_of(tensor)}, wanted {wanted_text} — attention works on plain 2-D (T, d) tensors here"
    if tuple(tensor.shape) != wanted:
        return False, f"got {shape_of(tensor)}, wanted {wanted_text}"
    return True, f"{shape_of(tensor)}"


def _rows_sum_to_one(fn) -> tuple[bool, str]:
    weights = as_tensor(fn(randn(6, 8, seed=13), randn(6, 8, seed=14)))
    if weights is None or weights.dim() != 2:
        return False, f"got {shape_of(weights if weights is not None else fn.__name__)}, cannot sum its rows"
    sums = weights.sum(dim=-1)
    worst = (sums - 1).abs().max().item()
    row = int((sums - 1).abs().argmax().item())
    return worst <= ATOL, f"row {row} sums to {sums[row].item():.6f}, wanted 1 (worst error {worst:.2e})"


def _single_token(weights) -> tuple[bool, str]:
    tensor = as_tensor(weights)
    if tensor is None or tuple(tensor.shape) != (1, 1):
        return False, f"got {shape_of(tensor) if tensor is not None else type(weights).__name__}, wanted (1 × 1)"
    return abs(tensor.item() - 1.0) <= ATOL, f"got [[{tensor.item():.6f}]], wanted [[1]]"


def _convex(fn) -> tuple[bool, str]:
    Q, K, V = randn(4, 6, seed=17), randn(4, 6, seed=18), randn(4, 5, seed=19)
    out = as_tensor(fn(Q, K, V))
    if out is None or tuple(out.shape) != (4, 5):
        return False, f"got {shape_of(out)}, wanted (4 × 5) before this check can run"
    low, high = V.min(dim=0).values, V.max(dim=0).values
    overshoot = max(0.0, float(torch.maximum(low - out, out - high).max().item()))
    return overshoot <= ATOL, f"a weighted average can never leave the range of the rows it averages (worst overshoot {overshoot:.2e})"


# --------------------------------------------------------------------- figures

TOKENS = ["the", "trophy", "fit", "it"]
Q = torch.tensor([[0.2, 0.9], [0.9, 0.2], [0.6, 0.4], [1.0, 0.0]])
K = torch.tensor([[0.2, 0.9], [1.0, 0.1], [0.6, 0.5], [0.3, 0.7]])
V = torch.tensor([[0.1, 0.1, 0.1], [1.0, 0.2, 0.0], [0.0, 0.6, 0.4], [0.2, 0.2, 0.6]])


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
    return {"source": "learner", "values": out.tolist()}


def figures(namespace) -> dict:
    """Numbers for the lesson page: whatever the learner's code produces right now."""
    return {
        "tokens": TOKENS,
        "dk": int(Q.shape[-1]),
        "weights": _run(_entry(namespace, "attention_weights"), (Q, K)),
        "output": _run(_entry(namespace, "attention"), (Q, K, V)),
    }
