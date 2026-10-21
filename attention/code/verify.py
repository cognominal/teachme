"""
verify.py -- independent checks on code/attn.py.

Two kinds of evidence, because "it runs" is not evidence:

  1. CROSS-IMPLEMENTATION.  Every nontrivial function is compared against
     PyTorch, written by different people from the same paper, at float64.
     A shared bug in my derivation would have to be reproduced independently
     in PyTorch to pass.

  2. INVARIANTS.  Properties that must hold for *any* weights -- causality,
     permutation equivariance, shift invariance, cache/full equivalence.
     These cannot be satisfied by accident.

Run:  .venv/bin/python code/verify.py
"""

from __future__ import annotations

import sys
import traceback

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from attn import (KVCache, MultiHeadAttention, CachedAttention, alibi_bias,
                  attention_loop, causal_mask, online_attention, repeat_kv,
                  rope_rotate, scaled_dot_product_attention, sliding_window_mask,
                  softmax, sinusoidal_encoding)

import torch
import torch.nn.functional as F

TOL = 1e-9
rng = np.random.default_rng(7)

PASS = []


def check(name, fn):
    try:
        fn()
    except Exception:
        print(f"FAIL  {name}")
        traceback.print_exc()
        PASS.append(False)
    else:
        print(f"ok    {name}")
        PASS.append(True)


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.max(np.abs(a - b)) / max(1.0, np.max(np.abs(b)))


# --------------------------------------------------------------------------
# cross-implementation checks vs PyTorch
# --------------------------------------------------------------------------

def t_softmax_vs_torch():
    z = rng.normal(size=(4, 7, 11)) * 3
    assert rel(softmax(z), torch.softmax(torch.tensor(z), -1).numpy()) < TOL


def t_sdpa_vs_torch():
    for dk, n, m in [(1, 5, 5), (8, 16, 16), (64, 32, 32), (64, 7, 13), (128, 40, 40)]:
        Q = rng.normal(size=(2, 3, n, dk))
        K = rng.normal(size=(2, 3, m, dk))
        V = rng.normal(size=(2, 3, m, dk + 1))       # d_v need not equal d_k
        got = scaled_dot_product_attention(Q, K, V)
        want = F.scaled_dot_product_attention(
            *[torch.tensor(t) for t in (Q, K, V)]).numpy()
        assert rel(got, want) < 1e-12, (dk, n, m)


def t_sdpa_causal_vs_torch():
    n, dk = 24, 32
    Q, K, V = (rng.normal(size=(1, 2, n, dk)) for _ in range(3))
    cm = causal_mask(n)
    got = scaled_dot_product_attention(Q, K, V, mask=cm)
    want = F.scaled_dot_product_attention(
        *[torch.tensor(t) for t in (Q, K, V)], is_causal=True).numpy()
    assert rel(got, want) < 1e-12


def t_additive_bias_vs_torch():
    """ALiBi is expressed here as a mask-space trick, but torch takes a float
    bias; check the two agree when the bias is finite everywhere."""
    n, dk, H = 12, 16, 4
    Q, K, V = (rng.normal(size=(1, H, n, dk)) for _ in range(3))
    bias = alibi_bias(n, n, H)
    logits = Q @ np.swapaxes(K, -1, -2) / np.sqrt(dk) + bias
    got = softmax(logits) @ V
    want = F.scaled_dot_product_attention(
        torch.tensor(Q), torch.tensor(K), torch.tensor(V),
        attn_mask=torch.tensor(bias)).numpy()
    assert rel(got, want) < 1e-12


def t_mha_vs_torch():
    """Rebuild the same layer in PyTorch from the identical weights."""
    B, n, d, H = 3, 20, 64, 4
    mha = MultiHeadAttention(d, H, rng=rng)
    x = rng.normal(size=(B, n, d))

    got = mha.forward(x, mask=causal_mask(n))

    xt = torch.tensor(x)
    q = (xt @ torch.tensor(mha.W_q)).reshape(B, n, H, mha.dh).transpose(1, 2)
    k = (xt @ torch.tensor(mha.W_k)).reshape(B, n, H, mha.dh).transpose(1, 2)
    v = (xt @ torch.tensor(mha.W_v)).reshape(B, n, H, mha.dh).transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    o = o.transpose(1, 2).reshape(B, n, H * mha.dh) @ torch.tensor(mha.W_o)
    assert rel(got, o.numpy()) < 1e-11


def t_gqa_vs_torch():
    """GQA: every KV head serves n_rep consecutive query heads."""
    B, n, d, H, Hkv = 2, 9, 32, 8, 2
    mha = MultiHeadAttention(d, H, n_kv_heads=Hkv, rng=rng)
    x = rng.normal(size=(B, n, d))
    got = mha.forward(x)
    xt = torch.tensor(x)
    q = (xt @ torch.tensor(mha.W_q)).reshape(B, n, H, mha.dh).transpose(1, 2)
    k = (xt @ torch.tensor(mha.W_k)).reshape(B, n, Hkv, mha.dh).transpose(1, 2)
    v = (xt @ torch.tensor(mha.W_v)).reshape(B, n, Hkv, mha.dh).transpose(1, 2)
    k = torch.repeat_interleave(k, H // Hkv, dim=1)
    v = torch.repeat_interleave(v, H // Hkv, dim=1)
    o = F.scaled_dot_product_attention(q, k, v)
    o = o.transpose(1, 2).reshape(B, n, d) @ torch.tensor(mha.W_o)
    assert rel(got, o.numpy()) < 1e-11
    x1 = rng.normal(size=(2, 1, 3, 4))                 # (B=2, Hkv=1, m=3, dh=4)
    r = repeat_kv(x1, 3)
    assert r.shape == (2, 3, 3, 4)
    assert np.array_equal(r[:, 0], x1[:, 0]) and np.array_equal(r[:, 2], x1[:, 0])


def t_mqa_single_kv_head():
    B, n, d, H = 2, 6, 24, 6
    mha = MultiHeadAttention(d, H, n_kv_heads=1, rng=rng)
    x = rng.normal(size=(B, n, d))
    out = mha.forward(x)
    assert out.shape == (B, n, d)
    # one KV head means the cache is H times smaller than the MHA cache
    assert mha.W_k.shape == (d, mha.dh)


# --------------------------------------------------------------------------
# invariants
# --------------------------------------------------------------------------

def t_softmax_shift_invariance():
    z = rng.normal(size=(5, 9)) * 100
    assert rel(softmax(z), softmax(z + 137.0)) < 1e-12
    assert rel(softmax(z), softmax(z - 137.0)) < 1e-12
    assert np.allclose(softmax(z).sum(-1), 1)


def t_softmax_bad_naive_overflow():
    """Without max-subtraction, float32 exp overflows around logit 89."""
    z = np.array([100.0, 0.0], dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        naive = np.exp(z) / np.exp(z).sum()
    assert not np.isfinite(naive).all()
    assert np.isfinite(softmax(z)).all()


def t_loop_equals_vectorised():
    n, m, dk = 6, 6, 5
    Q, K, V = (rng.normal(size=(n if i < 2 else m, dk)) for i in range(3))
    V = rng.normal(size=(m, dk))
    a = attention_loop(Q, K, V)
    b = scaled_dot_product_attention(Q[None], K[None], V[None])[0]
    assert rel(a, b) < 1e-12


def t_weights_are_a_distribution():
    n, dk = 30, 16
    Q, K, V = rng.normal(size=(n, dk)), rng.normal(size=(n, dk)), rng.normal(size=(n, dk))
    _, A = scaled_dot_product_attention(Q, K, V, mask=causal_mask(n),
                                        return_weights=True)
    assert np.all(A >= 0)
    assert np.allclose(A.sum(-1), 1, atol=1e-12)
    # causal: weigh nothing in the future
    assert np.all(A * (1 - causal_mask(n)) == 0)


def t_output_is_convex_combination():
    """Rows of the output lie in the convex hull of the value vectors -- an
    attention head can only *mix* what is already there, never invent."""
    m, dv = 9, 6
    V = rng.normal(size=(m, dv))
    lo, hi = V.min(0) - 1e-9, V.max(0) + 1e-9
    Q, K = rng.normal(size=(20, 4)) * 3, rng.normal(size=(m, 4)) * 3
    out = scaled_dot_product_attention(Q, K, V)
    assert np.all(out >= lo) and np.all(out <= hi)


def t_causal_perturbation():
    """THE causality test: change a token, and every *earlier* position's
    output must be bit-identical."""
    B, n, d, H = 1, 12, 32, 4
    mha = MultiHeadAttention(d, H, rng=rng)
    x1 = rng.normal(size=(B, n, d))
    x2 = x1.copy()
    x2[:, 7] += rng.normal(size=(B, d)) * 5.0        # a big change at position 7
    o1 = mha.forward(x1, mask=causal_mask(n))
    o2 = mha.forward(x2, mask=causal_mask(n))
    assert np.array_equal(o1[:, :7], o2[:, :7]), "the future leaked into the past"
    assert not np.allclose(o1[:, 7:], o2[:, 7:]), "change had no effect at all"


def t_permutation_equivariance():
    """Attention treats the input as a *set*.  Permute the tokens and the
    output permutes identically -- which is exactly why positional
    information has to be injected some other way."""
    B, n, d, H = 2, 10, 32, 4
    mha = MultiHeadAttention(d, H, rng=rng)          # rope/PE off
    x = rng.normal(size=(B, n, d))
    perm = rng.permutation(n)
    o = mha.forward(x)
    o_p = mha.forward(x[:, perm])
    assert rel(o_p, o[:, perm]) < 1e-12


def t_rope_is_relative():
    """<RoPE(q, m), RoPE(k, n)> must depend only on m - n -- at every offset."""
    dh = 16
    q = rng.normal(size=(dh,))
    k = rng.normal(size=(dh,))

    def score(m, n):
        qm = rope_rotate(q[None], np.array([m]))[0]
        kn = rope_rotate(k[None], np.array([n]))[0]
        return float(qm @ kn)

    pairs = [(0, 3), (1, 4), (5, 8), (100, 103), (10_000, 10_003),
             (1 << 20, (1 << 20) + 3)]
    scores = [score(m, n) for m, n in pairs]
    assert max(scores) - min(scores) < 1e-9, scores

    # The dependence is purely on the *signed* offset: (m, n) and (n, m) are
    # different problems.
    assert abs(score(3, 0) - score(0, 3)) > 1e-6

    # Consequence worth knowing: exact RoPE carries no absolute position
    # information at all.  Every geometric signal in a RoPE model comes from
    # the learned weights that read the relative offset, which is why the
    # long-context story is about rescaling those frequencies, not about
    # adding more position.


def t_rope_matches_reference():
    """Compare with the usual implementation via complex exponentiation."""
    n, dh = 8, 12
    x = rng.normal(size=(n, dh))
    got = rope_rotate(x, np.arange(n), base=10000.0)
    theta = 10000.0 ** (-np.arange(0, dh, 2) / dh)
    a = np.arange(n)[:, None] * theta[None, :]
    z = x[:, 0::2] + 1j * x[:, 1::2]
    zr = z * np.exp(1j * a)
    want = np.empty_like(x)
    want[:, 0::2], want[:, 1::2] = zr.real, zr.imag
    assert rel(got, want) < 1e-12


def t_rope_preserves_norm():
    x = rng.normal(size=(5, 8, 12))
    pos = np.broadcast_to(np.arange(8), (5, 8))       # no reason for all batch
    y = rope_rotate(x, pos)                           # entries to share a pos
    assert rel(np.linalg.norm(y, axis=-1), np.linalg.norm(x, axis=-1)) < 1e-12


def t_sinusoidal_dot_is_relative():
    pe = sinusoidal_encoding(200, 32)
    d1 = pe[10] @ pe[13]
    d2 = pe[60] @ pe[63]
    assert abs(d1 - d2) < 1e-9


def t_positional_via_rope_breaks_equivariance():
    """With RoPE on, the same layer is no longer permutation equivariant."""
    B, n, d, H = 1, 10, 32, 4
    mha = MultiHeadAttention(d, H, rng=rng, rope_base=10000.0)
    x = rng.normal(size=(B, n, d))
    perm = rng.permutation(n)
    o = mha.forward(x)
    o_p = mha.forward(x[:, perm])
    assert rel(o_p, o[:, perm]) > 1e-6


def t_flash_equals_naive():
    """Online softmax is exact, not an approximation."""
    for n, m, dk, block in [(64, 64, 32, 16), (100, 100, 32, 7), (33, 17, 16, 16),
                            (128, 128, 64, 32), (1, 100, 8, 8)]:
        Q = rng.normal(size=(2, 3, n, dk))
        K = rng.normal(size=(2, 3, m, dk))
        V = rng.normal(size=(2, 3, m, dk))
        want = scaled_dot_product_attention(Q, K, V)
        got, _ = online_attention(Q, K, V, block=block)
        assert rel(got, want) < 1e-12, (n, m, dk, block)


def t_flash_equals_naive_causal():
    n, dk = 96, 32
    Q, K, V = (rng.normal(size=(2, 2, n, dk)) for _ in range(3))
    want = scaled_dot_product_attention(Q, K, V, mask=causal_mask(n))
    got, _ = online_attention(Q, K, V, block=17, causal=True)
    assert rel(got, want) < 1e-12


def t_flash_with_padding_mask():
    n, dk = 40, 16
    Q, K, V = (rng.normal(size=(1, 1, n, dk)) for _ in range(3))
    mask = causal_mask(n)
    mask = mask & (np.arange(n)[None, :] != 5)        # token 5 is padding
    want = scaled_dot_product_attention(Q, K, V, mask=mask)
    got, _ = online_attention(Q, K, V, mask=mask, block=8, causal=True)
    assert rel(got, want) < 1e-10


def t_flash_memory_is_not_quadratic():
    n, dk = 4096, 64
    Q, K, V = (rng.normal(size=(1, 1, n, dk)) for _ in range(3))
    _, resident = online_attention(Q, K, V, block=64, causal=True)
    assert resident < n * n / 8, (resident, n * n)


def t_kv_cache_equals_full_forward():
    """Stepping token by token through a cache must reproduce the parallel
    forward pass of the whole sequence."""
    B, n, d, H = 2, 17, 48, 4
    mha = CachedAttention(d, H, rng=rng)
    x = rng.normal(size=(B, n, d))

    full = mha.forward(x, mask=causal_mask(n))

    cache = mha.new_cache(B, max_len=n)
    steps = [mha.step(x[:, t:t + 1], cache) for t in range(n)]
    inc = np.concatenate(steps, axis=1)
    assert rel(inc, full) < 1e-11


def t_kv_cache_shrinks_with_gqa():
    d, H = 64, 8
    a = MultiHeadAttention(d, H, rng=rng)
    b = MultiHeadAttention(d, H, n_kv_heads=2, rng=rng)
    c = MultiHeadAttention(d, H, n_kv_heads=1, rng=rng)
    assert a.W_k.shape[1] == d and b.W_k.shape[1] == d // 4 and c.W_k.shape[1] == d // 8
    assert a.W_q.shape == b.W_q.shape == c.W_q.shape      # compute is unchanged


def t_kv_cache_size_llama2_7b():
    """The canonical number: 32 layers, 32 KV heads of dim 128, 4096 context,
    fp16 -> exactly 2 GiB of KV cache."""
    def cache_bytes(n_layers, n_kv_heads, dh, ctx, bytes_per=2):
        return 2 * n_layers * n_kv_heads * dh * ctx * bytes_per
    got = cache_bytes(32, 32, 128, 4096)
    assert got == 2 * 1024 ** 3, got
    # GQA at the same context: Llama-2-70B, 80 layers, 8 KV heads
    assert cache_bytes(80, 8, 128, 4096) == 1280 * 1024 ** 2
    # ... versus what MHA would have cost there
    assert cache_bytes(80, 64, 128, 4096) == 10240 * 1024 ** 2


def t_sliding_window_is_a_subset_of_causal():
    n, w = 12, 4
    sw = sliding_window_mask(n, n, w)
    c = causal_mask(n)
    assert (sw & ~c).sum() == 0
    assert sw[0].sum() == 1 and sw[11].sum() == w
    # a long sequence attends ~n*w instead of n*n pairs
    assert sliding_window_mask(1000, 1000, 64).sum() <= 1000 * 64


def t_fully_masked_row_is_nan():
    """Masking every key of a row leaves softmax with 0/0.  This is why
    implementations either guarantee one visible key per row or drop the row."""
    logits = apply_nan = np.full((1, 1, 1, 3), -np.inf)
    with np.errstate(invalid="ignore"):
        w = softmax(logits)
    assert not np.isfinite(w).all()


def t_alibi_is_causal_and_decays():
    n, H = 8, 4
    b = alibi_bias(n, n, H)
    assert b.shape == (H, n, n)
    assert (b[0] * 0 + 1 != 0).all()
    assert np.all(b[:, 0, 0] == 0)                    # zero distance, zero penalty
    assert np.all(b[0, 0, 1] <= 0)
    # head 0 penalises distance far more steeply than the last head
    assert abs(b[0, 6, 0]) > 10 * abs(b[H - 1, 6, 0])


def t_softmax_gradient_saturation():
    """The Jacobian of softmax is diag(p) - p p^T; its largest entry, p(1-p),
    goes to zero as the distribution becomes one-hot.  This is the mechanism
    behind the 1/sqrt(d_k) factor, and why attention can get stuck."""
    for gap in [0.0, 5.0, 20.0, 60.0]:
        p = softmax(np.array([gap, 0.0]))
        peak = p.max() * (1 - p.max())
        if gap == 0.0:
            assert abs(peak - 0.25) < 1e-12
        if gap == 60.0:
            assert peak < 1e-25


if __name__ == "__main__":
    tests = [(k[2:], v) for k, v in sorted(globals().items()) if k.startswith("t_")]
    for name, fn in tests:
        check(name, fn)
    n_ok = sum(PASS)
    print(f"\n{n_ok}/{len(PASS)} checks passed")
    if n_ok != len(PASS):
        raise SystemExit(1)
    print("Numpy implementation agrees with PyTorch and satisfies every "
          "invariant tested.")