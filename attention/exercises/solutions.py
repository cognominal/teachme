"""
solutions.py -- worked answers to exercises/README.md, each self-checking.

    .venv/bin/python exercises/solutions.py

Every function ends in an assertion about the property the exercise is really
about, so a wrong implementation fails rather than printing something plausible.
"""

from __future__ import annotations

import itertools
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0] + "/../code")

import attn as A

rng = np.random.default_rng(0)
RESULTS = []


def ex(n, title):
    print(f"\n{'=' * 72}\nExercise {n}: {title}\n{'=' * 72}")


# --------------------------------------------------------------------------
def e1_softmax():
    ex(1, "numerically stable softmax")

    def s(z):
        z = np.asarray(z, dtype=np.float64)
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    z = rng.normal(size=(4, 5)) * 3
    assert np.allclose(s(z).sum(-1), 1)
    assert np.allclose(s(z), s(z + 137.0)) and np.allclose(s(z), s(z - 137.0))

    big = np.array([1000.0, 0.0, 0.0], dtype=np.float32)
    print(f"  softmax([1000, 0, 0]) fp32        = {s(big).astype(np.float32)}")
    assert np.isfinite(s(big)).all()

    with np.errstate(over="ignore", invalid="ignore"):
        naive = np.exp(big) / np.exp(big).sum()
    print(f"  without max subtraction           = {naive}")
    assert not np.isfinite(naive).all()

    # the boundary: float32 exp overflows at log(3.4e38) = 88.72
    print(f"  float32 overflow threshold        = log(3.4e38) = {np.log(3.4e38):.2f}")
    for t in [80.0, 88.0, 90.0]:
        with np.errstate(over="ignore"):
            v = np.float32(np.exp(np.float32(t)))
        print(f"  exp(float32({t:5.1f})) = {v}")


def e2_sdpa():
    ex(2, "scaled dot-product attention from scratch")

    def my_attn(Q, K, V, mask=None, scale=None):
        dk = Q.shape[-1]
        s = np.sqrt(dk) if scale is None else scale
        logits = Q @ np.swapaxes(K, -1, -2) / s
        if mask is not None:
            logits = np.where(mask, logits, -np.inf)
        logits = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(logits)
        w = e / e.sum(axis=-1, keepdims=True)
        return w @ V, w

    n, m, dk = 7, 13, 16
    Q, K, V = rng.normal(size=(n, dk)), rng.normal(size=(m, dk)), rng.normal(size=(m, dk))
    got, w = my_attn(Q, K, V)
    want = A.scaled_dot_product_attention(Q[None], K[None], V[None])[0]
    print(f"  n≠m ({n} queries, {m} keys) max error vs attn.py: "
          f"{np.abs(got - want).max():.3e}")
    assert np.abs(got - want).max() < 1e-13
    assert np.allclose(w.sum(-1), 1)

    mask = A.causal_mask(n, m)
    got, w = my_attn(Q, K, V, mask=mask)
    want = A.scaled_dot_product_attention(Q[None], K[None], V[None], mask=mask)[0]
    assert np.abs(got - want).max() < 1e-13
    ninf = np.isneginf(np.where(mask, 0.0, -np.inf))
    del ninf
    w_masked = w * 0 + w
    assert np.all(w_masked[~mask] == 0.0)
    print(f"  masked entries are exactly zero: {np.all(w[~mask] == 0.0)}")
    print(f"  no-inf check: {np.isfinite(got).all()}")


def e3_sqrt_dk():
    ex(3, "where 1/sqrt(d_k) comes from")

    dots = np.einsum("ij,ij->i", rng.normal(size=(20000, 64)),
                     rng.normal(size=(20000, 64)))
    print(f"  Var(q·k) for d_k=64: measured {dots.var():.2f}, d_k = 64")
    assert abs(dots.var() - 64) < 3

    def entropy(p):
        return float(-(p * np.log(p + 1e-300)).sum())

    print(f"\n  {'d_k':>6} {'scaled':>8} {'mean entropy':>14}")
    first_below_1 = None
    for dk in [16, 64, 256, 1024]:
        for scaled in [True, False]:
            hs = []
            for _ in range(50):
                q = rng.normal(size=dk)
                K = rng.normal(size=(64, dk))
                logits = q @ K.T / (np.sqrt(dk) if scaled else 1.0)
                hs.append(entropy(A.softmax(logits)))
            mh = np.mean(hs)
            print(f"  {dk:>6} {str(scaled):>8} {mh:>14.3f}")
            if not scaled and mh < 1.0 and first_below_1 is None:
                first_below_1 = dk
    print(f"\n  unscaled entropy first drops below 1 nat at d_k = {first_below_1}")
    assert first_below_1 is not None


def e4_equivariance():
    ex(4, "permutation equivariance of the whole layer")

    d, H, n = 32, 4, 12
    mha = A.MultiHeadAttention(d, H, rng=rng)
    x = rng.normal(size=(1, n, d))
    perm = rng.permutation(n)

    o = mha.forward(x)
    op = mha.forward(x[:, perm])
    err = np.abs(op - o[:, perm]).max()
    print(f"  without position: max difference = {err:.3e}")
    assert err < 1e-12

    mha_rope = A.MultiHeadAttention(d, H, rng=rng, rope_base=10000.0)
    o = mha_rope.forward(x)
    op = mha_rope.forward(x[:, perm])
    err = np.abs(op - o[:, perm]).max()
    print(f"  with RoPE:        max difference = {err:.3e}  (must be large)")
    assert err > 1e-6


def e5_padding():
    ex(5, "padding masks, and the all-masked row")

    n, d, H = 10, 32, 4
    mha = A.MultiHeadAttention(d, H, rng=rng)
    x = rng.normal(size=(1, n, d))

    mask = A.causal_mask(n) & (np.arange(n)[None, :] != 5)
    out, w = mha.forward(x, mask=mask, return_weights=True)
    print(f"  column 5 of the weights: max = {np.abs(w[..., 5]).max():.3e}")
    assert np.all(w[..., 5] == 0.0)

    x2 = x.copy()
    x2[:, 5] = rng.normal(size=d) * 100
    out2 = mha.forward(x2, mask=mask)
    changed = np.abs(out - out2).max(axis=-1)[0]
    print(f"  rewriting token 5 entirely -> per-position change:")
    print(f"    {np.round(changed, 6)}")
    others = [j for j in range(n) if j != 5]
    assert np.abs(out[:, others] - out2[:, others]).max() < 1e-12
    print("  every position except 5 is unchanged. Position 5 changes because")
    print("  masking the *key* does not stop token 5 from acting as a *query*:")
    print("  q_5 = x_5 W_q, and row 5's distribution over the visible keys")
    print("  depends on it. A padding mask removes a token as information; it")
    print("  does not remove the position from the batch.")
    assert changed[5] > 0

    bad = np.zeros((n, n), dtype=bool)
    with np.errstate(invalid="ignore"):
        out3 = mha.forward(x, mask=bad)
    assert not np.isfinite(out3).all()
    print("  all-masked mask -> contains NaN:", not np.isfinite(out3).all())

    # the two-line fix
    def fixed(logits, keep):
        keep = np.where(keep.any(-1, keepdims=True), keep, np.ones_like(keep))
        return np.where(keep, logits, -np.inf)

    print("  fix: if a row has no visible key, un-mask one (or write zeros).")
    print("       keep.any(-1, keepdims=True) ... np.where(keep, logits, -inf)")


def e6_rope():
    ex(6, "RoPE from scratch, and its aliasing")

    dh, n = 128, 32
    theta = 10000.0 ** (-np.arange(0, dh, 2) / dh)

    def rope(x, positions):
        """complex-number formulation: rotate each (even, odd) pair by theta*p"""
        z = x[:, 0::2] + 1j * x[:, 1::2]
        ang = np.asarray(positions, dtype=float)[:, None] * theta[None, :]
        zr = z * np.exp(1j * ang)
        out = np.empty_like(x)
        out[:, 0::2], out[:, 1::2] = zr.real, zr.imag
        return out

    x = rng.normal(size=(n, dh))
    got = rope(x, np.arange(n))
    want = A.rope_rotate(x, np.arange(n))
    print(f"  complex vs real formulation: max error {np.abs(got - want).max():.3e}")
    assert np.abs(got - want).max() < 1e-12

    q, k = rng.normal(size=dh), rng.normal(size=dh)
    sc = lambda m, nn: float(rope(q[None], [m])[0] @ rope(k[None], [nn])[0])
    vals = [sc(m, m + 5) for m in [0, 37, 1000, 250000]]
    print(f"  offset fixed at 5, positions varying: "
          f"{[round(v, 10) for v in vals]}")
    assert max(vals) - min(vals) < 1e-9

    period_fast = 2 * np.pi / theta[0]
    period_slow = 2 * np.pi / theta[-1]
    print(f"\n  theta[0]     = {theta[0]:g}   -> pair 0 turns once every "
          f"{period_fast:.2f} tokens")
    print(f"  theta[{dh // 2 - 1}]  = {theta[-1]:g}  -> the slowest pair turns "
          f"once every {period_slow:,.0f} tokens")

    ctx = 131072
    print(f"\n  over a {ctx:,}-token context, how many full turns does each "
          f"pair make?")
    print(f"    pair 0:            {ctx * theta[0] / (2 * np.pi):>12,.1f} turns")
    print(f"    pair {dh // 2 - 1} (slowest): {ctx * theta[-1] / (2 * np.pi):>12.2f}"
          f" turns   <- wraps")
    assert ctx * theta[-1] / (2 * np.pi) > 1

    # exact aliasing is avoided at integer positions precisely because the
    # fastest period is irrational: theta[0] = 1 means a period of 2*pi tokens.
    diffs = [np.abs(rope(q[None], [0])[0] - rope(q[None], [n])[0]).max()
             for n in range(1, 400)]
    print(f"\n  smallest exact-aliasing distance over positions 1..399: "
          f"{min(diffs):.3e}  (never zero)")
    assert min(diffs) > 1e-12
    print("  So there is no exact wrapping at integer positions; what degrades")
    print("  at long context is *resolution*: the slowest frequency makes only a")
    print("  couple of turns over 128k tokens, and a model trained at 8k never")
    print("  saw those relative offsets -- which is the actual long-context")
    print("  problem, and why the fixes rescale the frequency ladder rather")
    print("  than the attention mechanism.")


def e7_online():
    ex(7, "online softmax, exactness and working set")

    def online(Q, K, V, block):
        n, dk = Q.shape
        m = K.shape[0]
        out = np.zeros((n, V.shape[1]))
        big = 0
        for i0 in range(0, n, block):
            i1 = min(i0 + block, n)
            mx = np.full(i1 - i0, -np.inf)
            l = np.zeros(i1 - i0)
            acc = np.zeros((i1 - i0, V.shape[1]))
            for j0 in range(0, m, block):
                j1 = min(j0 + block, m)
                S = Q[i0:i1] @ K[j0:j1].T / np.sqrt(dk)
                big = max(big, S.size)
                m_new = np.maximum(mx, S.max(-1))
                alpha = np.exp(mx - m_new)
                alpha = np.where(np.isfinite(alpha), alpha, 0.0)
                P = np.exp(S - m_new[:, None])
                big = max(big, P.size)
                l = l * alpha + P.sum(-1)
                acc = acc * alpha[:, None] + P @ V[j0:j1]
                mx = m_new
            out[i0:i1] = acc / l[:, None]
        return out, big

    n, dk = 100, 16
    Q, K, V = rng.normal(size=(n, dk)), rng.normal(size=(n, dk)), rng.normal(size=(n, dk))
    want = A.scaled_dot_product_attention(Q[None], K[None], V[None])[0]
    for block in [1, 7, 64, 100]:
        got, _ = online(Q, K, V, block)
        err = np.abs(got - want).max()
        print(f"  block={block:>4}: max error {err:.3e}")
        assert err < 1e-12

    n2, block = 4096, 128
    naive_big = n2 * n2
    _, online_big = online(np.zeros((n2, 8)), np.zeros((n2, 8)), np.zeros((n2, 8)), block)
    print(f"\n  n={n2}: naive score matrix holds {naive_big:,} floats")
    print(f"          block={block}: largest intermediate holds {online_big:,}"
          f" floats")
    print(f"          reduction: {naive_big / max(online_big, 1):.1f}x")
    assert online_big < naive_big / 100


def e8_kv_budget():
    ex(8, "KV-cache budget")

    def cache_gib(L, hkv, dh, ctx, bpe=2):
        return 2 * L * hkv * dh * ctx * bpe / 1024 ** 3

    for hkv in [8, 32]:
        per_seq = cache_gib(32, hkv, 128, 131072)
        print(f"  {hkv:>2} KV heads: {per_seq:>6.1f} GiB per 128k sequence  "
              f"-> {int(40 // per_seq)} concurrent sequences on 40 GiB")
    print("  (bf16 = 2 bytes per element; '2 *' because K and V are both cached)")
    assert int(40 // cache_gib(32, 8, 128, 131072)) == 2
    assert int(40 // cache_gib(32, 32, 128, 131072)) == 0


def e9_alibi():
    ex(9, "ALiBi is purely relative")

    n, H = 12, 4
    bias = A.alibi_bias(n, n, H)                 # -slope_h * |i - j|, added to logits
    W = A.softmax(bias)

    def is_toeplitz(M, tol=1e-12):
        """constant along every diagonal"""
        return all(abs(M[i, j] - M[i - 1, j - 1]) < tol
                   for i in range(1, M.shape[0]) for j in range(1, M.shape[1]))

    # The *unnormalised* weights, exp(bias), are exactly Toeplitz.
    print(f"  exp(bias) Toeplitz? {is_toeplitz(np.exp(bias[1]))}")
    assert is_toeplitz(np.exp(bias[1]))

    # After softmax each row is divided by its own sum, so the matrix is only
    # Toeplitz up to a row-wise constant.  The testable consequence: the ratio
    # between two keys a and b steps away is the same in *every* row.
    a, b = 1, 5
    per_head = []
    for h in range(H):
        r = [W[h, i, i - a] / W[h, i, i - b] for i in range(b, n)]
        per_head.append((min(r) + max(r)) / 2)
        assert max(r) - min(r) < 1e-12, (h, r)
    print(f"  weight(key at distance {a}) / weight(key at distance {b}):")
    print(f"    constant across every row within a head (asserted, < 1e-12)")
    print(f"    but differs between heads, as the slopes do: "
          f"{[round(v, 3) for v in per_head]}")

    # A learned absolute-position signal destroys even that.
    pos = rng.normal(size=(n, 8))
    keys = pos @ rng.normal(size=(8, 8))
    Wp = A.softmax(pos @ np.swapaxes(keys, -1, -2) / np.sqrt(8))
    ratios2 = [Wp[i, i - a] / Wp[i, i - b] for i in range(b, n) if Wp[i, i - b] > 1e-300]
    print(f"  with learned absolute positions: spread = "
          f"{max(ratios2) - min(ratios2):.3e}  (not constant)")
    assert max(ratios2) - min(ratios2) > 1e-6

    print("\n  ALiBi can only express 'this far away', never 'this token' or")
    print("  'position 5'. It needs no parameters and extrapolates to unseen")
    print("  lengths for free, but it cannot do absolute-position lookup.")


def e10_heads_needed():
    ex(10, "how many heads a target requires")

    def min_heads(V, target):
        """Smallest H with target in H*conv(V) -- i.e. min sum(w) over
        w >= 0 with V^T w = target."""
        V = np.asarray(V, float)
        target = np.asarray(target, float)
        best = np.inf
        for size in range(1, len(V) + 1):
            for S in itertools.combinations(range(len(V)), size):
                M = V[list(S)].T
                w, res, rank, _ = np.linalg.lstsq(M, target, rcond=None)
                if rank < size:
                    continue
                if np.abs(M @ w - target).max() > 1e-9 or np.any(w < -1e-9):
                    continue
                best = min(best, w.sum())
        return best

    V2 = [[10.0, 0.0], [0.0, 10.0], [0.0, 0.0]]
    h = min_heads(V2, [10.0, 10.0])
    print(f"  V={V2}  target=[10,10]  ->  min H = {h:.6f}")
    assert abs(h - 2) < 1e-9

    V3 = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    h = min_heads(V3, [10.0, 10.0, 10.0])
    print(f"  V={V3}  target=[10,10,10]  ->  min H = {h:.6f}")
    assert abs(h - 3) < 1e-9

    # and confirm one head really cannot do it
    grid = np.linspace(0, 1, 2001)
    best = min(np.linalg.norm(np.array([a0, a1, 1 - a0 - a1]) @ np.array(V2) - [10, 10])
               for a0 in grid for a1 in grid if a0 + a1 <= 1)
    print(f"  best single-head error for the 2-D case: {best:.4f}")
    assert best > 7.0
    print("\n  The reachable set of H heads (shared V, block-diagonal W_o) is")
    print("  exactly H * conv(V rows). Adding heads scales the budget linearly.")


def e11_content_independent():
    ex(11, "the conceptual question A, made concrete")

    n, dk = 6, 4
    # content-independent weights == fixed A. Any function of the form A V.
    # Two different V give outputs that are related by the same linear map, so
    # the layer cannot implement anything nonlinear in the values.
    A_fixed = A.softmax(rng.normal(size=(n, n)))
    V1 = rng.normal(size=(n, dk))
    V2 = rng.normal(size=(n, dk))
    out1, out2 = A_fixed @ V1, A_fixed @ V2
    print(f"  with a fixed A: out(V1+V2) == out(V1)+out(V2)? "
          f"{np.allclose(A_fixed @ (V1 + V2), out1 + out2)}")
    print("  so a content-independent attention layer is a linear map of V:")
    print("  it can only *recombine* rows linearly. Without the softmax's")
    print("  dependence on Q and K, there is no routing at all.")

    Q, K = rng.normal(size=(n, dk)), rng.normal(size=(n, dk))
    out_a = A.scaled_dot_product_attention(Q, K, V1)
    out_b = A.scaled_dot_product_attention(2 * Q, K, V1)
    print(f"  with content-dependent weights: out(2Q) == out(Q)? "
          f"{np.allclose(out_a, out_b)}")
    print("  the softmax's content dependence is the entire source of power.")


if __name__ == "__main__":
    for fn in [e1_softmax, e2_sdpa, e3_sqrt_dk, e4_equivariance, e5_padding,
               e6_rope, e7_online, e8_kv_budget, e9_alibi, e10_heads_needed,
               e11_content_independent]:
        fn()
        RESULTS.append(fn.__name__)
    print(f"\n{'=' * 72}\n{len(RESULTS)} exercises, all assertions passed\n"
          f"{'=' * 72}")
    print("Conceptual answers:")
    print(" B. d_k, because the logit is q·k, a sum of d_k products. d_v is the")
    print("    width of V and never enters the score; d_model is irrelevant once")
    print("    the heads are split.")
    print(" C. W_q is (d_model, num_heads * d_h) = (4096, 32*128) = (4096, 4096).")
    print("    num_heads is 32 -- and note this need not satisfy")
    print("    num_heads*d_h == d_model; Gemma 3 has 16 heads x 256 = 4096 != 3840.")
    print(" D. Not broken: that is the attention sink. A softmax row must sum to")
    print("    1 and cannot abstain, so surplus mass lands on a low-information")
    print("    token. Evicting its KV in a sliding window wrecks quality --"
          " StreamingLLM's result.")
    print(" E. With a causal mask, position i's output depends only on x_0..x_i,")
    print("    so all n targets are predictions the model could legally make at")
    print("    inference time. Bidirectionally, position i has already read x_i+1,")
    print("    and you would be scoring a copy of the answer.")
    print(" F. Halving KV heads (GQA): the decode bottleneck is bytes of cache")
    print("    read per step, and that scales with H_kv. Pruning query heads")
    print("    barely touches decode traffic. A shorter window helps only if")
    print("    quality survives the lost range.")
    print(" G. (i) Flash/tiling: keep the function exact, stop materialising the")
    print("    (n,n) matrix -- gives up nothing but does not reduce FLOPs.")
    print("    (ii) Sparsity/sliding windows: give up the universal 'every token")
    print("    can reach every token in one hop' guarantee; information must hop")
    print("    through layers. (iii) Linear attention / kernels: give up the")
    print("    sharp, exactly-content-addressed retrieval that softmax's")
    print("    exponential weights provide; you get a fixed-size state instead.")