"""
attn.py -- attention, implemented from scratch in NumPy.

Every function here is a direct transcription of the equations in ../notes/.
No autograd, no framework, no hidden magic: read it top to bottom and you know
exactly what an attention layer computes.

Shape conventions used throughout
---------------------------------
    B    batch size
    H    number of query heads
    Hkv  number of key/value heads (Hkv < H == grouped-query attention)
    n    number of query positions            (n_q)
    m    number of key/value positions        (n_kv);  m == n for self-attention
    d    d_model, the width of the residual stream
    dh   head dimension;  dh == d // H

    Q : (B, H,   n, dh)
    K : (B, Hkv, m, dh)
    V : (B, Hkv, m, dh)
    A : (B, H,   n, m)    -- the attention weight matrix, rows sum to 1

Mask convention: a mask is a boolean array broadcastable to (..., n, m) where
True means "this key is visible to this query" and False means "forbidden".
PyTorch's functional API uses the same convention for its `attn_mask`.
"""

from __future__ import annotations

import numpy as np

NEG_INF = -np.inf


# ===========================================================================
# 1. softmax -- the only nonlinearity inside attention
# ===========================================================================

def softmax(z, axis=-1):
    """Numerically stable softmax.

    softmax(z)_i = exp(z_i) / sum_j exp(z_j)

    Two facts you will use constantly:

    * Shift invariance.  softmax(z + c) == softmax(z) for any scalar c, because
      the constant cancels between numerator and denominator.  We exploit this
      by subtracting the row max, which is exact, not an approximation -- and
      necessary, because exp(90) overflows float32.

    * It overwrites magnitude with ratio.  Adding 1.0 to every logit changes
      nothing.  Only *differences* matter, so the logits are a scale-free
      ranking of keys, and their overall scale (the reason for the 1/sqrt(d_k))
      is the knob that controls how peaked that ranking becomes.
    """
    z = np.asarray(z)
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


# ===========================================================================
# 2. the attention mechanism itself
# ===========================================================================

def apply_mask(logits, mask):
    """Set forbidden positions to -inf so softmax gives them exactly 0 weight."""
    return np.where(mask, logits, NEG_INF)


def scaled_dot_product_attention(Q, K, V, mask=None, scale=None,
                                 return_weights=False):
    """Attn(Q, K, V) = softmax(Q K^T / sqrt(d_k) + M) V

    Q : (..., n, dk)
    K : (..., m, dk)
    V : (..., m, dv)
    mask : (..., n, m) bool, True == visible,  None == every key visible

    returns (..., n, dv)  [and the weight matrix if return_weights=True]
    """
    dk = Q.shape[-1]
    s = np.sqrt(dk) if scale is None else scale

    # (..., n, dk) @ (..., dk, m) -> (..., n, m).  This is the ONLY place where
    # two different tokens talk to each other.
    logits = Q @ np.swapaxes(K, -1, -2)
    logits = logits / s

    if mask is not None:
        logits = apply_mask(logits, mask)

    A = softmax(logits, axis=-1)          # (..., n, m)
    out = A @ V                           # (..., n, dv)
    return (out, A) if return_weights else out


def attention_loop(Q, K, V):
    """Same math as scaled_dot_product_attention, written with three explicit
    Python loops and no broadcasting at all.  This exists only to be slow and
    obvious; lab.py checks that it agrees with the vectorised version to the
    last bit.  Read this one first if the fancy version is opaque.

    Q, K, V : (n, dk), (m, dk), (m, dv)   -- a single head, single sequence.
    """
    n, dk = Q.shape
    m, dv = V.shape
    out = np.zeros((n, dv))
    for i in range(n):                    # for each query position i
        scores = np.zeros(m)
        for j in range(m):                # compare query i to every key j
            scores[j] = np.dot(Q[i], K[j]) / np.sqrt(dk)
        weights = softmax(scores)         # (m,) non-negative, sums to 1
        for j in range(m):                # weighted average of the values
            out[i] += weights[j] * V[j]
    return out


# ===========================================================================
# 3. masks
# ===========================================================================

def causal_mask(n_q, n_kv=None, offset=0):
    """Lower-triangular 'look backwards only' mask.

    mask[i, j] is True iff key j is at or before query i.

    `offset` is the number of positions that the query block is displaced by,
    i.e. the number of cached tokens when doing incremental decoding.  With
    offset=0 you get the usual (n, n) lower-triangular mask; with offset=k the
    query at row i is global position k+i and may see keys 0..k+i.
    """
    n_kv = n_q if n_kv is None else n_kv
    qi = np.arange(offset, offset + n_q)[:, None]      # (n_q, 1)
    kj = np.arange(n_kv)[None, :]                      # (1, n_kv)
    return kj <= qi


def sliding_window_mask(n_q, n_kv, window, offset=0):
    """Causal + at most `window` tokens of history (Mistral / Longformer style).
    mask[i, j] is True iff offset+i-window < j <= offset+i."""
    qi = np.arange(offset, offset + n_q)[:, None]
    kj = np.arange(n_kv)[None, :]
    return (kj <= qi) & (kj > qi - window)


def alibi_bias(n_q, n_kv, n_heads, offset=0, base=2.0 ** -8):
    """ALiBi: no positional encoding at all, just a per-head linear penalty on
    distance, added to the logits before softmax.  Head h gets slope
    base**h, so head 0 decays fast and the last head is almost unmasked.
    Returns a bias of shape (n_heads, n_q, n_kv) to be *added* to the logits."""
    qi = np.arange(offset, offset + n_q)[:, None]
    kj = np.arange(n_kv)[None, :]
    dist = (qi - kj)                               # (n_q, n_kv), >= 0 if causal
    slopes = base ** np.arange(n_heads)            # (H,)
    return -np.abs(dist)[None, :, :] * slopes[:, None, None]


def repeat_kv(x, n_rep):
    """Grouped-query attention: expand Hkv KV heads to H query heads by
    repeating each KV head n_rep = H // Hkv times.  (..., Hkv, m, dh) -> (..., H, m, dh)"""
    if n_rep == 1:
        return x
    return np.repeat(x, n_rep, axis=-3)


# ===========================================================================
# 4. multi-head attention
# ===========================================================================

def xavier(rng, shape, gain=1.0):
    """A sane random init: keeps activation variance roughly constant across a
    layer, which is why trained transformers don't start out saturated."""
    fan_in, fan_out = shape[-2], shape[-1]
    limit = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape)


class MultiHeadAttention:
    """The whole thing: four projection matrices and H independent attention
    operations.

        head_i = Attn(x W_q_i, x W_k_i, x W_v_i)
        out    = concat(head_1 ... head_H) W_o

    Parameter count: 3 * H * (d * dh) + d * d = 4 d^2 when dh = d/H.

    `n_kv_heads` implements grouped-query attention (GQA): if it is smaller
    than `n_heads`, K and V are projected to fewer heads and each KV head is
    shared by `n_heads // n_kv_heads` query heads.  Set n_kv_heads == n_heads
    (default) for classic MHA, or n_kv_heads == 1 for MQA.
    """

    def __init__(self, d_model, n_heads, n_kv_heads=None, rng=None,
                 dtype=np.float64, rope_base=None):
        rng = np.random.default_rng(0) if rng is None else rng
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        if n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.n_rep = n_heads // self.n_kv_heads
        self.dh = d_model // n_heads
        self.dtype = dtype
        self.rope_base = rope_base          # None == no rotary embedding

        mk = lambda a, b: xavier(rng, (a, b)).astype(dtype)
        self.W_q = mk(d_model, n_heads * self.dh)
        self.W_k = mk(d_model, self.n_kv_heads * self.dh)
        self.W_v = mk(d_model, self.n_kv_heads * self.dh)
        self.W_o = mk(n_heads * self.dh, d_model)

    # -- projections ------------------------------------------------------
    def qkv(self, x, positions=None):
        """x: (B, n, d) -> q, k, v each (B, H*, n, dh).

        Note the head split: the d_model axis is cut into n_heads chunks of
        size dh, and each chunk becomes one head.  The heads are not special
        subspaces chosen by hand -- they are just slices of the projection
        output, and any partition would be equivalent given the freedom in
        W_q/W_k/W_v.
        """
        B, n, _ = x.shape
        q = (x @ self.W_q).reshape(B, n, self.n_heads, self.dh).transpose(0, 2, 1, 3)
        k = (x @ self.W_k).reshape(B, n, self.n_kv_heads, self.dh).transpose(0, 2, 1, 3)
        v = (x @ self.W_v).reshape(B, n, self.n_kv_heads, self.dh).transpose(0, 2, 1, 3)
        if self.rope_base is not None:
            if positions is None:
                positions = np.arange(n)
            q = rope_rotate(q, positions, self.rope_base)
            k = rope_rotate(k, positions, self.rope_base)
        return q, k, v

    # -- forward ----------------------------------------------------------
    def forward(self, x, mask=None, positions=None, return_weights=False):
        """x: (B, n, d) -> (B, n, d)."""
        B, n, _ = x.shape
        q, k, v = self.qkv(x, positions)
        k = repeat_kv(k, self.n_rep)        # (B, H, n, dh)
        v = repeat_kv(v, self.n_rep)
        out, A = scaled_dot_product_attention(q, k, v, mask=mask,
                                              return_weights=True)
        out = out.transpose(0, 2, 1, 3).reshape(B, n, self.n_heads * self.dh)
        out = out @ self.W_o
        return (out, A) if return_weights else out


# ===========================================================================
# 5. KV cache -- what makes generation O(1) per token instead of O(n)
# ===========================================================================

class KVCache:
    """Preallocated K/V storage for autoregressive decoding.

    Training feeds the whole sequence at once and costs O(n^2).  Generation
    feeds one token at a time; without a cache you would recompute K and V for
    every past token at every step (n^2/2 work in total per sequence).  With
    the cache you store K and V once per position and each step costs O(n).

    Memory cost: 2 * n_layers * n_kv_heads * dh * n_ctx * bytes_per_element.
    """

    def __init__(self, batch, n_kv_heads, max_len, dh, dtype=np.float64):
        self.k = np.zeros((batch, n_kv_heads, max_len, dh), dtype=dtype)
        self.v = np.zeros((batch, n_kv_heads, max_len, dh), dtype=dtype)
        self.n = 0

    def append(self, k_new, v_new):
        """k_new, v_new: (B, Hkv, t, dh).  Returns the full K and V so far."""
        t = k_new.shape[2]
        if self.n + t > self.k.shape[2]:
            raise RuntimeError("KV cache overflow")
        self.k[:, :, self.n:self.n + t] = k_new
        self.v[:, :, self.n:self.n + t] = v_new
        self.n += t
        return self.k[:, :, :self.n], self.v[:, :, :self.n]


class CachedAttention(MultiHeadAttention):
    """MHA that can be stepped one token at a time against a KV cache."""

    def new_cache(self, batch, max_len):
        return KVCache(batch, self.n_kv_heads, max_len, self.dh, self.dtype)

    def step(self, x, cache, return_weights=False):
        """x: (B, 1, d) -- the newest token only.  Appends to cache, attends
        over everything cached so far."""
        B, t, _ = x.shape
        n_seen = cache.n
        q, k, v = self.qkv(x, positions=np.arange(n_seen, n_seen + t))
        K, V = cache.append(k, v)                      # (B,Hkv,n_seen+t,dh)
        K = repeat_kv(K, self.n_rep)
        V = repeat_kv(V, self.n_rep)
        # With one query per step and a full cache, causality is automatic:
        # the query is at position n_seen+t-1 and every cached key precedes it.
        out, A = scaled_dot_product_attention(q, K, V, return_weights=True)
        out = out.transpose(0, 2, 1, 3).reshape(B, t, self.n_heads * self.dh)
        out = out @ self.W_o
        return (out, A) if return_weights else out


# ===========================================================================
# 6. ranking positions: sinusoidal, rotary, learned
# ===========================================================================

def sinusoidal_encoding(n, d, base=10000.0):
    """The original Transformer positional encoding (Vaswani et al. 2017).

    PE[p, 2i]   = sin(p / base^(2i/d))
    PE[p, 2i+1] = cos(p / base^(2i/d))

    Added to the input embeddings.  Wavelengths span 2*pi to 2*pi*base, so
    dimensions form a geometric "clock" from fast to slow.  The dot product
    of two such encodings is a function of p-q, which is what makes it
    relative-ish -- but only through the *added* signal, so it interacts with
    the learned weights rather than being exactly relative."""
    pos = np.arange(n)[:, None]                       # (n, 1)
    i = np.arange(0, d, 2)[None, :]                   # (1, d/2)
    ang = pos / base ** (i / d)
    pe = np.zeros((n, d))
    pe[:, 0::2] = np.sin(ang)
    pe[:, 1::2] = np.cos(ang)
    return pe


def _rope_freqs(dh, base):
    """theta_i = base^(-2i/dh) for i = 0 .. dh/2-1."""
    i = np.arange(0, dh, 2)
    return base ** (-i / dh)


def rope_rotate(x, positions, base=10000.0):
    """Rotary position embedding (RoPE), Su et al. 2021.

    Treat each (even, odd) pair of head-dimension channels as a 2-D point and
    rotate it by an angle proportional to the token's position:

        [x'_2i  ]   [cos(p*theta_i)  -sin(p*theta_i)] [x_2i  ]
        [x'_2i+1] = [sin(p*theta_i)   cos(p*theta_i)] [x_2i+1]

    Applied to q and k only (never v), *after* the projections.  The payoff:
    <RoPE(q, m), RoPE(k, n)> depends only on m-n, so the dot product -- which
    is the only thing attention sees -- becomes a function of *relative*
    offset rather than absolute position.

    x         : (..., n, dh), dh even
    positions : broadcastable to x.shape[:-1], e.g. (n,) for any batch shape
    """
    dh = x.shape[-1]
    if dh % 2:
        raise ValueError("RoPE needs an even head dimension")
    theta = _rope_freqs(dh, base)                            # (dh/2,)
    ang = np.asarray(positions)[..., None] * theta           # broadcastable to
    #                                                          (..., n, dh/2)
    cos, sin = np.cos(ang), np.sin(ang)

    xe, xo = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = xe * cos - xo * sin
    out[..., 1::2] = xe * sin + xo * cos
    return out


# ===========================================================================
# 7. flash attention: online softmax, exact and memory-frugal
# ===========================================================================

def online_attention(Q, K, V, mask=None, block=64, causal=False):
    """Attention computed exactly, without ever materialising the (n, m)
    score matrix, using the 'online softmax' trick.

    The plain algorithm needs the whole row of scores before it can compute
    its normaliser.  Flash attention instead walks the key axis in blocks,
    keeping three running accumulators:

        m   running maximum score seen so far
        l   running sum of exp(score - m)
        acc running sum of exp(score - m) * v

    When a new block raises the maximum from m_old to m_new, everything
    accumulated so far is multiplied by exp(m_old - m_new) -- which rescales
    the old partial softmax numerator and denominator by exactly the same
    factor, so the ratio acc/l is unchanged.  The result is bit-comparable to
    the naive version up to floating-point rounding, not an approximation.

    Memory: O(n * block) instead of O(n * m).  That, plus keeping the traffic
    in SRAM instead of round-tripping through HBM, is the entire reason flash
    attention is faster.  It computes the same function.

    Supports an arbitrary (..., n, m) boolean mask, and skips blocks that lie
    entirely in the future when causal=True.
    """
    Q, K, V = np.asarray(Q), np.asarray(K), np.asarray(V)
    *lead, n, dh = Q.shape
    m_kv = K.shape[-2]
    scale = 1.0 / np.sqrt(dh)

    out = np.empty((*lead, n, V.shape[-1]), dtype=Q.dtype)
    max_resident = max(block * block, block * V.shape[-1])

    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        Qi = Q[..., i0:i1, :]                                  # (..., bq, dh)
        mi = np.full((*lead, i1 - i0), -np.inf)                # running max
        li = np.zeros((*lead, i1 - i0))                        # running denom
        acc = np.zeros((*lead, i1 - i0, V.shape[-1]))          # running numerator

        for j0 in range(0, m_kv, block):
            if causal and j0 >= i1:
                break                     # this whole KV block is in the future
            j1 = min(j0 + block, m_kv)
            Kj = K[..., j0:j1, :]
            S = (Qi @ np.swapaxes(Kj, -1, -2)) * scale         # (..., bq, bk)
            if mask is not None:
                S = apply_mask(S, mask[..., i0:i1, j0:j1])
            if causal:
                rows = np.arange(i0, i1)[:, None]
                cols = np.arange(j0, j1)[None, :]
                S = apply_mask(S, cols <= rows)

            m_new = np.maximum(mi, np.max(S, axis=-1))
            m_new = np.where(np.isfinite(m_new), m_new, 0.0)
            alpha = np.exp(mi - m_new)                         # rescale factor
            alpha = np.where(np.isfinite(alpha), alpha, 0.0)
            P = np.exp(S - m_new[..., None])
            li = li * alpha + np.sum(P, axis=-1)
            acc = acc * alpha[..., None] + P @ V[..., j0:j1, :]
            mi = m_new

        out[..., i0:i1, :] = acc / li[..., None]
        del S, P, alpha, m_new        # (conceptually: these never leave SRAM)

    return out, max_resident