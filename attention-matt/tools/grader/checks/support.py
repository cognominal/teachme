"""support.py — the small vocabulary every lesson's checks share."""

from __future__ import annotations

import math

import torch

# Float32 chains (matmul + softmax) land within ~4e-8 of the same maths in
# float64 — measured in this workspace on 2026-10-21 — so 1e-5 is seven orders
# of slack for a correct implementation and still far tighter than any of the
# mistakes these checks diagnose.
ATOL = 1e-5


def randn(*shape: int, seed: int) -> torch.Tensor:
    """Reproducible test data: the same numbers on every run, in every front end."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=generator)


def as_tensor(value) -> torch.Tensor | None:
    """Accept a tensor, or a plain nested list that torch can read as one."""
    if isinstance(value, torch.Tensor):
        return value
    try:
        return torch.as_tensor(value, dtype=torch.float32)
    except (TypeError, ValueError, RuntimeError):
        return None


def shape_of(value) -> str:
    tensor = as_tensor(value)
    if tensor is None:
        return type(value).__name__
    return "(" + " × ".join(str(d) for d in tensor.shape) + ")"


def describe(value) -> str:
    tensor = as_tensor(value)
    if tensor is None:
        return repr(value)[:60]
    if tensor.numel() <= 16:
        flat = ", ".join(f"{x:.4f}" for x in tensor.reshape(-1).tolist())
        return f"{shape_of(tensor)} [{flat}]"
    return f"{shape_of(tensor)} tensor"


def max_diff(a, b) -> float | None:
    """Largest absolute difference, or None when the two cannot be compared."""
    ta, tb = as_tensor(a), as_tensor(b)
    if ta is None or tb is None or tuple(ta.shape) != tuple(tb.shape):
        return None
    return (ta.double() - tb.double()).abs().max().item()


def near(a, b, atol: float = ATOL) -> bool:
    """True when `a` is numerically the same as `b` (and both exist)."""
    diff = max_diff(a, b)
    return diff is not None and diff <= atol


def match_detail(a, b, atol: float = ATOL) -> tuple[bool, str]:
    diff = max_diff(a, b)
    if diff is None:
        return False, f"got {shape_of(a)}, wanted {shape_of(b)}"
    if diff == 0.0:
        return True, "identical to the reference"
    return diff <= atol, f"largest cell differs from the reference by {diff:.2e}"


def scaled_scores(Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    return Q @ K.transpose(-2, -1) / math.sqrt(K.shape[-1])


def row_sums(t: torch.Tensor) -> list[float]:
    return [round(x, 8) for x in t.sum(dim=-1).tolist()]
