"""
lab.py -- runnable demonstrations.  Each lab prints numbers, and most also
assert the property they are demonstrating, so you can trust the output.

    .venv/bin/python code/lab.py            # all labs
    .venv/bin/python code/lab.py scale rope # a subset

Read notes/*.md alongside; every lab says which section it belongs to.
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from attn import (MultiHeadAttention, CachedAttention, alibi_bias, attention_loop,
                  causal_mask, online_attention, repeat_kv, rope_rotate,
                  scaled_dot_product_attention, sliding_window_mask, softmax,
                  sinusoidal_encoding)

RNG = np.random.default_rng(0)


def rule(title, section=""):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")
    if section:
        print(f"(notes: {section})\n")


def _time(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


# =========================================================================
def lab_softmax():
    rule("LAB 1 -- softmax is a soft argmax, and it only sees ratios",
         "01-mechanics.md 1.2")

    z = np.array([2.0, 1.0, 0.0, -1.0])
    print(f"logits           {z}")
    print(f"softmax          {np.round(softmax(z), 4)}")
    print(f"softmax(z + 10)  {np.round(softmax(z + 10), 4)}   <- add 10 to every logit")
    print(f"softmax(3 * z)   {np.round(softmax(3 * z), 4)}   <- multiply by 3")
    print("Adding a constant changes nothing: only differences matter.")
    print("Multiplying changes everything: the *scale* is the temperature.")
    assert np.allclose(softmax(z), softmax(z + 10))

    print("\nAs the gap between the top logit and the rest widens:")
    print(f"{'gap':>6} {'top weight':>12} {'rest weight':>12} {'entropy':>9}"
          f" {'grad p(1-p)':>12}")
    for gap in [0, 1, 2, 4, 8, 16, 32]:
        p = softmax(np.array([gap, 0.0, 0.0, 0.0]))
        ent = -(p * np.log(p + 1e-300)).sum()
        print(f"{gap:>6} {p[0]:>12.6f} {p[1]:>12.6f} {ent:>9.4f} "
              f"{p.max() * (1 - p.max()):>12.3e}")
    print("At gap 32 the distribution is effectively one-hot and p(1-p), the")
    print("softmax Jacobian's largest entry, is ~1e-14: that head can no longer")
    print("learn from that example.  This is what the 1/sqrt(d_k) fights.")


# =========================================================================
def lab_scale():
    rule("LAB 2 -- why divide by sqrt(d_k)", "01-mechanics.md 1.3")

    print("If q and k have independent, mean-0, unit-variance entries, then")
    print("q . k = sum_i q_i k_i has mean 0 and variance d_k.\n")
    print(f"{'d_k':>5} {'std(q.k) measured':>18} {'sqrt(d_k)':>10}")
    for dk in [1, 4, 16, 64, 256, 1024]:
        Q = RNG.normal(size=(2000, dk))
        K = RNG.normal(size=(2000, dk))
        dots = np.einsum("ij,ij->i", Q, K)
        print(f"{dk:>5} {dots.std():>18.3f} {np.sqrt(dk):>10.2f}")
        assert abs(dots.std() - np.sqrt(dk)) < 0.1 * np.sqrt(dk)

    print("\nAnd what that does to the attention weights (m = 64 keys, "
          "averaged over 200 draws):")
    m, trials = 64, 200
    print(f"  {'d_k':>5} {'scaled?':>8} {'std(logit)':>11} {'max weight':>11}"
          f" {'entropy':>8} {'max grad':>10}")
    for dk in [16, 64, 256]:
        for scaled in [False, True]:
            stds, maxes, ents, grads = [], [], [], []
            for _ in range(trials):
                Q, K = RNG.normal(size=dk), RNG.normal(size=(m, dk))
                logits = (Q @ K.T) / (np.sqrt(dk) if scaled else 1.0)
                p = softmax(logits)
                stds.append(logits.std())
                maxes.append(p.max())
                ents.append(-(p * np.log(p + 1e-300)).sum())
                grads.append(p.max() * (1 - p.max()))
            print(f"  {dk:>5} {str(scaled):>8} {np.mean(stds):>11.3f} "
                  f"{np.mean(maxes):>11.4f} {np.mean(ents):>8.3f} "
                  f"{np.mean(grads):>10.2e}")
    print(f"\n  entropy of a uniform row over {m} keys = {np.log(m):.3f} (the maximum).")
    print("  Unscaled, the logits widen with sqrt(d_k) and the distribution")
    print("  collapses toward one-hot: the entropy falls, and p(1-p) -- the")
    print("  largest entry of the softmax Jacobian -- falls with it, so there")
    print("  is almost no gradient left to learn from. Dividing by sqrt(d_k)")
    print("  holds logit std at ~1 for every head width, which is exactly the")
    print("  condition under which a fresh, randomly initialised head can")
    print("  still move.")


# =========================================================================
def lab_shapes():
    rule("LAB 3 -- the shapes, once and for all", "01-mechanics.md 1.4")

    B, n, d, H = 2, 5, 8, 2
    dh = d // H
    mha = MultiHeadAttention(d, H, rng=RNG)
    x = RNG.normal(size=(B, n, d))
    print(f"x  input          {x.shape}   (batch, seq, d_model)")
    q, k, v = mha.qkv(x)
    print(f"q  queries        {q.shape}   (batch, heads, seq, d_head)")
    print(f"k  keys           {k.shape}")
    print(f"v  values         {v.shape}")
    scores = q @ np.swapaxes(k, -1, -2)
    print(f"scores q k^T      {scores.shape}   (batch, heads, n_q, n_kv)"
          "   <- the quadratic part")
    A = softmax(scores / np.sqrt(dh))
    print(f"attention weights {A.shape}   rows sum to {A.sum(-1).mean():.6f}")
    ctx = A @ v
    print(f"context  A v      {ctx.shape}   (batch, heads, n_q, d_head)")
    out = ctx.transpose(0, 2, 1, 3).reshape(B, n, d)
    print(f"concat heads      {out.shape}   (batch, seq, n_heads * d_head)")
    out = out @ mha.W_o
    print(f"W_o               {out.shape}   (batch, seq, d_model)  <- back in")
    print("\nOne line to remember: (n, d_model) -> (H, n, d_head) ->")
    print("(H, n, n) weights -> (H, n, d_head) -> (n, d_model).")
    print("Only the (H, n, n) tensor mixes tokens; everything else is per-token.")


# =========================================================================
def lab_three_loops():
    rule("LAB 4 -- the textbook loop, and why nobody writes it",
         "01-mechanics.md 1.1")

    n, dk = 4, 3
    Q = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0], [1.0, 1.0, 1.0]])
    K = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    V = np.array([[10.0, 0.0], [0.0, 10.0], [5.0, 5.0], [1.0, 1.0]])

    slow = attention_loop(Q, K, V)
    fast = scaled_dot_product_attention(Q[None], K[None], V[None])[0]
    print("Q (4 queries x d_k=3), K (4 keys), V (4 values x 2):\n")
    for i in range(n):
        print(f"query {i}: scores = Q[{i}] . K[j] / sqrt(3) = "
              f"{np.round(Q[i] @ K.T / np.sqrt(dk), 3)}")
        print(f"          weights = {np.round(softmax(Q[i] @ K.T / np.sqrt(dk)), 3)}"
              f"  ->  output = {np.round(slow[i], 4)}")
    print("\nthree nested loops :", np.round(slow, 10).tolist())
    print("one matmul chain   :", np.round(fast, 10).tolist())
    print("identical:", np.allclose(slow, fast))
    print("\nThe loop is what 'attention' means.  The three matmuls are the same")
    print("thing with the batch and head axes folded in -- nothing more.")
    assert np.allclose(slow, fast)


# =========================================================================
def lab_causal():
    rule("LAB 5 -- causality: the mask, and the reason transformers generate",
         "01-mechanics.md 1.5")

    n, d, H = 10, 32, 4
    mha = MultiHeadAttention(d, H, rng=RNG)
    x = RNG.normal(size=(1, n, d))

    print("causal_mask(6):")
    print(causal_mask(6).astype(int))
    print("\nTrue == visible. Note the row-0 query sees exactly one key.")
    print("A later query (row 9) sees ten.  Attention is therefore *not*")
    print("uniformly powerful along the sequence.\n")

    o1 = mha.forward(x, mask=causal_mask(n))
    x2 = x.copy()
    x2[:, 6] += RNG.normal(size=(1, d)) * 100.0          # wreck one token
    o2 = mha.forward(x2, mask=causal_mask(n))

    changed = np.abs(o1 - o2).max(axis=-1)[0]
    print("perturb positions >= 6 (a huge change at position 6):")
    print("per-position output change:", np.round(changed, 6))
    print("\npositions 0..5 are bit-identical; positions 6..9 all move.")
    assert np.array_equal(o1[:, :6], o2[:, :6])
    assert np.all(changed[6:] > 0)

    print("\nNow the same without the mask (a bidirectional encoder):")
    o3 = mha.forward(x)
    o4 = mha.forward(x2)
    print("per-position output change:", np.round(np.abs(o3 - o4).max(-1)[0], 6))
    print("every position moves, including the ones *before* the edit.")
    print("That is why an encoder can be bidirectional and a decoder cannot.")

    print("\nSpecial case -- a row where every key is masked:")
    logits = np.full((3,), -np.inf)
    print("  logits ", logits)
    with np.errstate(invalid="ignore"):
        print("  softmax", softmax(logits.astype(float)))
    print("  Every entry of the row is 0 and the denominator is 0: not a")
    print("  distribution, a NaN. This bites for real with fully-padded")
    print("  sequences and with rows whose whole visible window was evicted.")
    print("  Implementations avoid it by guaranteeing at least one visible")
    print("  key per row (an explicit 'always visible' sink token is the")
    print("  usual trick -- see LAB 12).")


# =========================================================================
def lab_position():
    rule("LAB 6 -- attention is a set operation, so position must be injected",
         "01-mechanics.md 1.6 / 03-in-the-transformer.md 3.3")

    n, d, H = 8, 32, 4
    mha = MultiHeadAttention(d, H, rng=RNG)
    x = RNG.normal(size=(1, n, d))
    perm = np.array([3, 0, 7, 1, 5, 2, 6, 4])
    o, A = mha.forward(x, return_weights=True)
    op, Ap = mha.forward(x[:, perm], return_weights=True)
    print("no positional information:")
    print(f"  max |out(permuted tokens) - permuted out(tokens)| = "
          f"{np.abs(op - o[:, perm]).max():.3e}")
    print("  i.e. the outputs are exactly the multiset of the original")
    print("  outputs, just reordered. The computation cannot tell which")
    print("  token came first: 'a then b' and 'b then a' are the same")
    print("  function. Undo the permutation on the weights and they match")
    print("  to the last bit:")
    print(f"  max |A[perm, perm] - A_permuted| = "
          f"{np.abs(A[:, :, perm][:, :, :, perm] - Ap).max():.3e}")
    print("  Yet within a single query, the weights are content-dependent:")
    print(f"    row 0 attends with {np.round(A[0, 0, 0], 3)} -- and a")
    print("    different token occupies each slot after the shuffle. Attention")
    print("    matches *content*, never position. Position must come from")
    print("    outside (PE, RoPE, ALiBi, or a masking pattern).")

    pe = sinusoidal_encoding(n, d)
    print(f"\nsinusoidal PE {pe.shape} -- first 6 dims of rows 0,1,2:")
    print(np.round(pe[:3, :6], 3))
    print(f"  PE[0] . PE[3] = {pe[0] @ pe[3]:.6f}")
    print(f"  PE[60] . PE[63] = {sinusoidal_encoding(200, d)[60] @ sinusoidal_encoding(200, d)[63]:.6f}"
          "   <- same offset, same dot product")

    print("\nRoPE: rotate q and k by an angle proportional to position.")
    dh = 8
    q, k = RNG.normal(size=dh), RNG.normal(size=dh)

    def rscore(m, n):
        return float(rope_rotate(q[None], np.array([m]))[0]
                     @ rope_rotate(k[None], np.array([n]))[0])

    print(f"  <rope(q,0),   rope(k,3)>   = {rscore(0, 3):.6f}")
    print(f"  <rope(q,5),   rope(k,8)>   = {rscore(5, 8):.6f}")
    print(f"  <rope(q,100), rope(k,103)> = {rscore(100, 103):.6f}")
    print(f"  <rope(q,1M),  rope(k,1M+3)> = {rscore(1 << 20, (1 << 20) + 3):.6f}")
    print("  same offset => same score, at ANY distance. That is the trick:")
    print("  the dot product -- the only thing attention sees -- is a function")
    print("  of relative offset and nothing else.")
    print("  Norm is preserved (it is a rotation), so logit scale is preserved:")
    print(f"  |q|={np.linalg.norm(q):.6f}  "
          f"|rope(q,12345)|={np.linalg.norm(rope_rotate(q[None], np.array([12345]))[0]):.6f}")


# =========================================================================
def lab_multihead():
    rule("LAB 7 -- why more than one head", "02-multi-head.md 2.2")

    print("A head produces ONE weighted average per query: out = sum_j a_j V_j")
    print("with a_j >= 0 and sum a_j = 1. The output is therefore always inside")
    print("the convex hull of the value vectors. Let us make that bite.\n")

    # Three tokens. Token 1 holds [10, 0], token 2 holds [0, 10].
    # The query wants to copy BOTH coordinates.
    V = np.array([[10.0, 0.0], [0.0, 10.0], [0.0, 0.0]])
    want = np.array([10.0, 10.0])

    # best single-head attempt: search the simplex of attention weights
    best, best_a = np.inf, None
    grid = np.linspace(0, 1, 1001)
    for a0 in grid:
        for a1 in grid:
            if a0 + a1 > 1:
                continue
            a = np.array([a0, a1, 1 - a0 - a1])
            d = np.linalg.norm(a @ V - want)
            if d < best:
                best, best_a = d, a
    print(f"target output                     {want}")
    print(f"values available                  {V.tolist()}")
    print(f"best single head: weights {np.round(best_a, 4)} -> "
          f"{np.round(best_a @ V, 4)}, error {best:.4f}")
    print("No weights can do it: reaching x=10 needs a_1=1, reaching y=10 needs")
    print("a_2=1, and a_1+a_2 <= 1. The deficit is not a training problem, it")
    print("is an algebraic one -- and it does not shrink with more data. A")
    print("second head is a second, independent softmax distribution.\n")

    # Two heads: two independent averages, each one free to pick a different
    # position.  K is made orthogonal so each head's target is unambiguous;
    # the softmax is still soft, so the selection is exact only in the limit
    # of a large logit gap -- watch the error collapse as q grows.
    K = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    print(f"  {'|q|':>5} {'head 1 weights (key 1)':>24} {'2-head output':>22}"
          f" {'error':>10}")
    for mag in [8.0, 16.0, 32.0, 64.0]:
        q1, q2 = np.array([mag, 0.0]), np.array([0.0, mag])
        a1 = softmax(q1 @ K.T / np.sqrt(2))
        a2 = softmax(q2 @ K.T / np.sqrt(2))
        out = a1 @ V + a2 @ V
        err = np.linalg.norm(out - want)
        print(f"  {mag:>5.0f} {a1[0]:>24.12f} "
              f"{str(np.round(out, 12)):>22} {err:>10.2e}")
    print("\n  Softmax is never exactly one-hot, but its error falls off like")
    print("  exp(-gap): an arbitrarily good approximation is always available.")
    print("  The single-head error, by contrast, is a hard 7.07.")

    q1, q2 = np.array([64.0, 0.0]), np.array([0.0, 64.0])
    out = sum(softmax(q @ K.T / np.sqrt(2)) @ V for q in (q1, q2))
    assert np.linalg.norm(out - want) < 1e-15

    print("\nThat is the whole argument: H heads give H independent softmax")
    print("distributions per query, recombined by W_o. The heads are not")
    print("`different subspaces' by design -- they are slices of the projection")
    print("output, free to specialise during training. And they do:")
    print("  * previous-token heads: attend to position i-1")
    print("  * induction heads: 'find the earlier occurrence of the current")
    print("    token and copy what followed it' -- the mechanism behind in-")
    print("    context learning (Olsson et al., 2022)")
    print("  * heads that attend almost entirely to the first token (see the")
    print("    attention-sink lab, LAB 11)")
    print("Measured, not assumed: run code/tiny_transformer.py to watch heads")
    print("split the work in a model you trained yourself in ~40 seconds.")


# =========================================================================
def lab_kv_cache():
    rule("LAB 8 -- training is parallel, generation is not", "03-in-the-transformer.md 3.5")

    B, n, d, H = 1, 12, 48, 4
    mha = CachedAttention(d, H, rng=RNG)
    x = RNG.normal(size=(B, n, d))

    t0 = time.perf_counter()
    full = mha.forward(x, mask=causal_mask(n))
    t_parallel = time.perf_counter() - t0

    cache = mha.new_cache(B, max_len=n)
    t0 = time.perf_counter()
    inc = np.concatenate([mha.step(x[:, t:t + 1], cache) for t in range(n)], axis=1)
    t_serial = time.perf_counter() - t0

    print(f"parallel forward of {n} tokens : {t_parallel * 1e3:7.3f} ms")
    print(f"{n} sequential cached steps     : {t_serial * 1e3:7.3f} ms")
    print(f"max difference                 : {np.abs(full - inc).max():.3e}")
    print("Identical outputs -- the cache is an optimisation, not an")
    print("approximation. What differs is *work*.\n")

    print("Cost of one K and one V projection per token: 2 d_model^2 FLOPs")
    print("(per layer).  Generating n tokens:")
    print(f"\n  {'n':>7} {'with cache':>16} {'no cache':>18} {'saved':>9}")
    for n2 in [128, 1024, 8192, 32768]:
        cached = 2 * d * d * n2                        # one projection each
        uncached = d * d * n2 * (n2 + 1)               # re-project 1..n tokens
        print(f"  {n2:>7} {cached:>16,} {uncached:>18,} "
              f"{uncached / cached:>8.0f}x")
    print("\nWithout a cache a naive decoder re-projects every past token at")
    print("every step: sum_t O(t) = O(n^2) projections for n tokens. With it,")
    print("exactly n projections. The attention itself -- the (n, n) matmuls")
    print("against the cached K and V -- is unchanged, and is what remains.")

    print("\nKV cache size = 2 * layers * n_kv_heads * d_head * ctx * bytes")
    print(f"{'model':<18} {'L':>3} {'Hkv':>4} {'dh':>4} {'ctx':>6} "
          f"{'fp16 cache':>12}")
    models = [
        ("GPT-2 small",     12, 12,  64,  1024),
        ("Llama-2-7B",      32, 32, 128,  4096),
        ("Llama-2-70B (GQA)", 80,  8, 128,  4096),
        ("Llama-2-70B (MHA)", 80, 64, 128,  4096),
        ("Llama-3-8B (GQA)", 32,  8, 128,  8192),
        ("Llama-3-8B @128k", 32,  8, 128, 131072),
    ]
    for name, L, hkv, dh, ctx in models:
        nbytes = 2 * L * hkv * dh * ctx * 2
        unit = nbytes / 1024 ** 2
        print(f"{name:<18} {L:>3} {hkv:>4} {dh:>4} {ctx:>6} {unit:>9.1f} MiB")
    print("\nA batch of 32 at 128k context is 32 x 16 GiB = 512 GiB of KV alone.")
    print("That single number drove MQA, GQA and MLA. See LAB 11.")


# =========================================================================
def lab_flash():
    rule("LAB 9 -- flash attention: same answer, no (n, n) matrix", "04-modern.md 4.2")

    n, dk = 2048, 64
    Q, K, V = (RNG.normal(size=(1, 1, n, dk)) for _ in range(3))

    t0 = time.perf_counter()
    naive, A = scaled_dot_product_attention(Q, K, V, mask=causal_mask(n),
                                            return_weights=True)
    t_naive = time.perf_counter() - t0

    t0 = time.perf_counter()
    online, resident = online_attention(Q, K, V, block=128, causal=True)
    t_fast = time.perf_counter() - t0

    print(f"n = {n}, d_head = {dk}")
    print(f"naive : {t_naive * 1e3:7.2f} ms,  score matrix "
          f"{A.nbytes / 1024 ** 2:7.2f} MiB allocated")
    print(f"blocked: {t_fast * 1e3:7.2f} ms,  largest working set "
          f"{resident * 8 / 1024:7.1f} KiB")
    print(f"max |naive - blocked| = {np.abs(naive - online).max():.3e}   "
          f"(exact, up to float rounding)")
    assert np.abs(naive - online).max() < 1e-10

    print("\nThe trick is one line of algebra. Let p_j = exp(s_j - m) be the")
    print("softmax numerator. If a new block raises the running max from m_old")
    print("to m_new, then for every previously seen j:")
    print("    exp(s_j - m_new) = exp(s_j - m_old) * exp(m_old - m_new)")
    print("so you multiply the accumulated numerator acc and denominator l by")
    print("the SAME factor, and acc/l is unchanged. You never need to see the")
    print("whole row before you can start accumulating. Underflow-safe, exact,")
    print("and it never materialises an (n, n) tensor.\n")
    assert np.all(np.isfinite(online))


# =========================================================================
def lab_cost():
    rule("LAB 10 -- quadratic in sequence length, linear in width", "04-modern.md 4.1")

    d, H = 256, 4
    mha = MultiHeadAttention(d, H, rng=RNG)
    print(f"d_model = {d}, {H} heads, float64, single process")
    print("timings are the best of 5 runs after a warmup -- a single run is")
    print("dominated by allocator and BLAS-setup noise\n")
    print(f"{'n':>6} {'attn ms':>9} {'ratio':>7} {'score matrix':>14}")
    prev = None
    for n in [256, 512, 1024, 2048, 4096]:
        x = RNG.normal(size=(1, n, d))
        mha.forward(x, mask=causal_mask(n))                    # warmup
        best = min(_time(lambda: mha.forward(x, mask=causal_mask(n)))
                   for _ in range(5))
        ratio = "-" if prev is None else f"{best / prev:.2f}"
        print(f"{n:>6} {best * 1e3:>9.2f} {ratio:>7} "
              f"{n * n * H * 8 / 1024 ** 2:>11.1f} MiB")
        prev = best
    print("Doubling n roughly quadruples the time and the memory: the O(n^2).")

    print("\nPer layer, for a sequence of n tokens (2 FLOPs per multiply-add):\n")
    print("  attention  =  4 n^2 d      (2n^2d for QK^T, 2n^2d for AV)")
    print("  MLP        = 16 n d^2      (2nd*4d up + 2n*4d*d down)")
    print(f"\n=> they cross at n = 4d = {4 * d} for this width.")
    print("Below that the feed-forward block dominates; above it, attention")
    print("does. Which is why long-context work is attention work, and why")
    print("every efficiency paper targets the (n, n) tensor rather than the")
    print("projections.")

    print("\nAlso worth internalising: attention has no parameters in the")
    print("mechanism itself. All the weights live in the four projections")
    print(f"(4 * d^2 = {4 * d * d:,} here). The attention operation is a fixed")
    print("routing function over whatever W_q, W_k, W_v happen to produce.")


# =========================================================================
def lab_variants():
    rule("LAB 11 -- MQA / GQA / MLA / sliding window / ALiBi", "04-modern.md")

    d, H = 4096, 32
    dh = d // H
    print("KV cache per token per layer, d_model=4096, 32 query heads, "
          "head_dim=128:\n")
    rows = [
        ("MHA", 32, "Hkv = 32 = H"),
        ("GQA (Llama-2-70B style)", 8, "8 groups of 4 query heads"),
        ("MQA", 1, "all heads share one KV"),
    ]
    for name, hkv, note in rows:
        elems = 2 * hkv * dh
        print(f"  {name:<26} {elems:>7,} numbers/token  "
              f"({hkv}/{H} of MHA)   {note}")
    print(f"  {'MLA (DeepSeek-V2, V3)':<26} {512 + 64:>7,} numbers/token"
          "   (1/28 of MHA)   low-rank latent + 64 RoPE dims")
    print("\nMQA/GQA shrink the cache by shrinking the *number of KV heads*: no")
    print("new idea, just fewer, larger groups. MLA shrinks it by caching a")
    print("low-rank latent and re-expanding K and V on the fly; the cache never")
    print("grows back because the up-projection is folded into W_q and W_o")

    print("\n--- sliding window (Mistral-style local attention) ---")
    n = 12
    mask = sliding_window_mask(n, n, window=4)
    print(mask.astype(int))
    print(f"pairs attended: {(mask).sum()} of {n * n} = "
          f"{100 * mask.sum() / n ** 2:.0f}%")
    n2, w = 1000, 64
    m2 = sliding_window_mask(n2, n2, w)
    print(f"at n={n2}, window={w}: {m2.sum():,} of {n2 ** 2:,} pairs "
          f"({100 * m2.sum() / n2 ** 2:.2f}%) -> O(n * w), linear in n")
    print("The information still travels further than w tokens, because it")
    print("hops through intermediate layers. Cost: O(n * w * layers).")

    print("\n--- ALiBi: no positional encoding at all ---")
    H2, n3 = 8, 10
    bias = alibi_bias(n3, n3, H2)
    w = softmax(bias[:, -1, :])
    print("attention weights of the LAST token, head 0 vs head 7:")
    print("  head 0:", np.round(w[0], 4), " <- local")
    print("  head 7:", np.round(w[7], 4), " <- nearly uniform")
    print("A per-head linear penalty on distance, added to the logits. No")
    print("parameters, no PE, and it extrapolates to unseen lengths -- but it")
    print("cannot represent 'attend to position 5'.")

    print("\n--- content vs position, the field's oldest argument ---")
    print("NoPE (no positional encoding at all) works surprisingly well for")
    print("decoder-only models: with causal masking, a head can already infer")
    print("'how many tokens back am I' from the *shape* of what it can see.")


# =========================================================================
def lab_attention_sink():
    rule("LAB 12 -- the attention sink, and why softmax rows must sum to 1",
         "04-modern.md 4.6")

    print("softmax has no way to say 'attend to nothing'. Every row is forced")
    print("to distribute exactly 1.0 of attention, so when a head has nothing")
    print("useful to look at it dumps the mass somewhere. Trained models")
    print("reliably dump it on the first token -- a token with no semantic")
    print("content at all. That is the 'attention sink' (Xiao et al., 2023).\n")

    # Demonstrate the pressure with a concrete model-free version of the claim.
    scores = RNG.normal(size=1000) * 4          # 1000 keys, all irrelevant
    p = softmax(scores)
    print(f"1000 irrelevant keys, logit std 4:")
    print(f"  largest weight {p.max():.4f}  on key {p.argmax()}")
    print(f"  even the WORST key gets {p.min():.3e} -- never zero")
    print(f"  sum = {p.sum():.6f}  (must be 1)")
    print("\nSo a head that wants 'no information' must still pick a winner.")
    print("Adding one always-visible 'null' key is one fix (softmax1, Miller")
    print("2023); bisecting the mass is another (Differential Transformer,")
    print("Ye et al. 2024). Both are ways of letting attention say 'nothing'.")

    print("\nConsequences you can check in any real model's attention maps:")
    print("  * a large share of heads put >50% of their mass on token 0")
    print("  * evicting the first token's KV from a sliding window destroys")
    print("    quality, hence StreamingLLM keeps it permanently")
    print("  * this is why 'sink tokens' appear even in vision and audio models")


# =========================================================================
def lab_hardware():
    rule("LAB 13 -- what actually costs time: bytes moved, not FLOPs",
         "04-modern.md 4.2")

    n, d = 2048, 4096
    print(f"n={n}, d_model={d}, fp16\n")
    print("Naive attention writes and re-reads the (n, n) score matrix, once")
    print("per layer:\n")
    H = 32
    scores = n * n * H * 2
    qkv = 3 * n * d * 2
    x = n * d * 2
    weights = 4 * d * d * 2
    print(f"  input activations x : {n} x {d} x 2 B            = "
          f"{x / 1024 ** 2:7.1f} MiB")
    print(f"  q, k, v activations : 3 x that                 = "
          f"{qkv / 1024 ** 2:7.1f} MiB")
    print(f"  attention weights (this layer) : 4 d^2 x 2 B    = "
          f"{weights / 1024 ** 2:7.1f} MiB")
    print(f"  score matrix A      : {n} x {n} x {H} x 2 B          = "
          f"{scores / 1024 ** 2:7.1f} MiB")
    print(f"\n  A is {scores / x:.1f}x the input it was computed FROM, and "
          f"{scores / weights:.1f}x this layer's")
    print("  entire set of attention weight matrices -- for one layer. It is")
    print("  written, read for the softmax, written again, read for A@V.")
    print("\nA100: ~19.5 TFLOP/s fp16 versus ~2 TB/s of HBM bandwidth. Moving a")
    print("byte costs roughly an order of magnitude more than the 100 or so")
    print("FLOPs you could do with it. A kernel that avoids the round trip wins")
    print("even if it does identical arithmetic -- which is exactly what flash")
    print("attention is: same FLOPs, ~10x fewer bytes moved.")
    print("\nThis is the general lesson of modern inference work: count memory")
    print("traffic first, FLOPs second. It is also why the KV cache -- pure")
    print("bytes, linear in context -- is the binding constraint in production.")


# =========================================================================
LABS = {k[4:]: v for k, v in list(globals().items()) if k.startswith("lab_")}

if __name__ == "__main__":
    which = sys.argv[1:] or list(LABS)
    for name in which:
        if name not in LABS:
            print(f"unknown lab {name!r}; available: {', '.join(LABS)}")
            raise SystemExit(2)
        LABS[name]()
    print()