# 01 — The mechanics

Everything in this file is implemented in `code/attn.py` and demonstrated, with
numbers, in `code/lab.py` (labs 1–6). Run them as you read.

---

## 0. The problem attention solves

Before 2017, a sequence model had two options.

**Recurrence.** Process one token at a time, carrying a hidden state `h_t`.
Every piece of information about the past has to fit in that one fixed-size
vector. The model must decide, at every step, what to keep — and that decision
is made before the model knows what it will need. Worse, it is sequential: step
`t` cannot start until step `t-1` finishes, so a 100k-token document means 100k
dependent steps.

**Convolution.** A fixed receptive field, so the number of layers needed to
relate two distant tokens grows with the distance between them.

**Attention** replaces both. Every position gets a direct, content-addressed
link to every other position, computed in a single parallel step. Nothing is
compressed into a fixed-size bottleneck; the model looks things up on demand.

```
for each position i:                       # n positions, all in parallel
    score_i = how well does position i match each other position j?
    weights = normalize(score_i)           # a distribution over positions
    output_i = weighted average of the *contents* of those positions
```

The price is `O(n²)` work: every position considers every other position. Every
efficiency paper in this space is an attempt to pay less while keeping the
lookup. That tradeoff is the subject of file 04.

One more framing, which is the one that generalises best: attention is a
**soft, differentiable dictionary lookup**.

| symbol | metaphor | what it is |
|---|---|---|
| `q_i` (query) | the search string | what position *i* is looking for |
| `k_j` (key) | the label on the entry | what position *j* advertises about itself |
| `v_j` (value) | the payload | what position *j* hands over if selected |

A hard lookup would be `output_i = V[argmax_j sim(q_i, k_j)]`. That is not
differentiable and gives no credit to near misses. Softening the `argmax` into
`softmax` makes it differentiable, and simultaneously makes it *composite*: the
model can read several entries at once, in any blend.

---

## 1.1 The definition, written as loops

With `Q ∈ R^{n×d_k}`, `K ∈ R^{m×d_k}`, `V ∈ R^{m×d_v}`:

```
scores  = Q Kᵀ / sqrt(d_k)          (n, m)      how well each query matches each key
A       = softmax(scores, dim=-1)   (n, m)      rows are distributions over keys
output  = A V                       (n, d_v)    each row is a weighted average
```

Spelled out as three nested loops (`attn.py::attention_loop`, demonstrated in
lab 4):

```python
for i in range(n):                          # each query
    for j in range(m):                      # each key
        s[j] = dot(Q[i], K[j]) / sqrt(d_k)  # similarity
    a = softmax(s)                          # a distribution: a >= 0, sum(a) == 1
    out[i] = sum(a[j] * V[j] for j in range(m))   # one weighted average
```

Later, as a one-liner:

```python
out = softmax(Q @ K.T / sqrt(d_k)) @ V
```

These are the same function. The loops are the definition; the matmuls are the
definition with the batch and head axes folded in. Lab 4 checks they agree to
the last bit.

> **The single most useful sentence in this file:** the output at position *i*
> is a **convex combination** of the value vectors. Attention can only *mix*
> what is already there. It cannot create information, only route it.

That has teeth. If your heads are `h`, each output row lives in the convex hull
of the rows of `V` — a bounded, low-dimensional region (lab 7 exploits exactly
this). Nothing about attention increases the rank or the "information content"
of the sequence; it only re-distributes it. Knowledge lives in the *parameters*
(the MLPs and the projections); attention is how that knowledge gets moved
between positions at inference time.

### Why a dot product?

The score must be a similarity between a query and a key, and it must be
learned. A bilinear form `qᵀ W k` is the simplest choice; by folding any
`W` into the projections (define `q = W_q x`, `k = W_k x`) the model gets an
arbitrary learned bilinear form for free. The dot product is then just the
cheapest way to evaluate a bilinear form whose matrix the model has already
absorbed.

The alternative — additive attention (`wᵀ tanh(W_q q + W_k k)`), used in the
original Bahdanau translation model — is a small MLP instead of a dot product.
It works, but it cannot be expressed as a single matmul, and it is not
expressible as one fused kernel. Dot-product attention won on hardware terms as
much as on modelling terms.

---

## 1.2 softmax: the only nonlinearity inside attention

```
softmax(z)_i = exp(z_i) / Σ_j exp(z_j)
```

Four properties matter for attention. Lab 1 measures all of them.

**1. Shift invariance.** `softmax(z + c) ≡ softmax(z)` exactly: the constant
cancels between numerator and denominator. We exploit this by subtracting the
row maximum before exponentiating. That is not an approximation — it is exact —
and it is *necessary*, because `exp(90)` already overflows float32 (`exp(88.7)`
is near the float32 ceiling of 3.4e38). Without the shift, a head with large
logits silently returns `inf/inf = nan`.

**2. Only ratios matter; absolute magnitude is a temperature.** Adding `1.0` to
every logit changes nothing. Doubling every logit is not a no-op: it sharpens
the distribution. "How sharp" is controlled by the *spread* of the logits, and
the entire purpose of the `1/sqrt(d_k)` factor is to keep that spread
approximately constant (§1.3). Conversely, the fact that a model's confidence
is a function of logit *differences* is why temperature sampling works when
decoding.

**3. Its Jacobian is `diag(p) - p pᵀ`.** The largest entry is `p(1-p) ≤ 0.25`,
attained at the uniform distribution. As the distribution approaches one-hot,
`p(1-p) → 0` and the gradient through that softmax vanishes. This is the
mechanism behind the saturation argument in §1.3, and it is why a head whose
logits are wildly spread can be effectively frozen: no gradient reaches its
Q/K projections.

**4. It cannot abstain.** A softmax row sums to exactly 1. There is no
"attend to nothing". Every head must distribute all of its weight, even when it
has nothing useful to look at. Trained models resolve this by dumping the
surplus onto some arbitrary token — usually the first token of the sequence,
a token with no semantic content at all. That is the *attention sink*, and it
has real consequences for eviction policies and quantization (file 04, §4.6).

---

## 1.3 Why divide by √d_k

Suppose the components of `q` and `k` are independent, mean 0, variance 1. Then

```
⟨q, k⟩ = Σ_{i=1}^{d_k} q_i k_i
E[⟨q,k⟩] = 0
Var(⟨q,k⟩) = Σ_i Var(q_i k_i) = Σ_i E[q_i²]E[k_i²] = d_k
```

so the logits have standard deviation `√d_k`. At `d_k = 256` that is a spread of
16 logits — and a softmax over logits that are ±16 apart is essentially one-hot
before training even starts. Note that the *rank* of the keys is unaffected by
scaling; only the sharpness changes. So is the argmax. But the sharpness is
what the gradient depends on (§1.2, point 3).

Dividing by `√d_k` rescales the spread back to ~1, independent of head width.
Lab 2 measures both halves: it verifies `Var(⟨q,k⟩) ≈ d_k`, then shows entropy
collapsing and `p(1-p)` collapsing with it, at three head widths, with and
without the scale.

**Caveats, because this is where textbooks stop and practice starts.**

- The variance argument assumes independence and unit variance, which trained
  models do not satisfy. Activations have heavy tails and are correlated across
  dimensions. The real reason `1/√d_k` survives is that it is *approximately*
  right and empirically robust, not that it is provably optimal.
- Modern models often add a second, more direct handle on logit scale:
  **QK-norm** (normalising `q` and `k` before the dot product, as in ViT-22B,
  Gemma 2/3, and many 2024+ models) or **logit soft-capping** (clamping logits
  to `±softcap` via a scaled `tanh`, as in Gemma 2). These bound the logit
  spread explicitly, which bounds the saturation directly — a cleaner fix for
  the same problem.
- `1/√d_k` sets the *initial* scale. A trained head can learn to make its
  logits as sharp as it likes through `W_q`/`W_k`. The factor is about giving
  optimisation a sane starting point, not about constraining the final
  solution.

---

## 1.4 Shapes, and the one tensor that matters

For a batch of `B` sequences, `H` heads, `n` queries, `m` keys and head
dimension `d_h = d_model / H` (printed by lab 3):

| tensor | shape | meaning |
|---|---|---|
| `x` | `(B, n, d_model)` | the residual stream |
| `q` | `(B, H, n, d_h)` | per-head queries |
| `k` | `(B, H, m, d_h)` | per-head keys |
| `v` | `(B, H, m, d_h)` | per-head values |
| `scores = q kᵀ` | `(B, H, n, m)` | **the quadratic tensor** |
| `A = softmax(scores/√d_h)` | `(B, H, n, m)` | rows sum to 1 |
| `A v` | `(B, H, n, d_h)` | per-head context |
| `concat + W_o` | `(B, n, d_model)` | back into the residual stream |

The head split is just a reshape: the `d_model` axis is cut into `H` contiguous
chunks of size `d_h`, and chunk `h` is head `h`. Heads are not special subspaces
chosen by design — any partition is equivalent given the freedom in
`W_q, W_k, W_v`.

Only the `(n, m)` tensor mixes tokens. Every other operation in the list is
per-position: `q_i` depends only on `x_i`. Keep that in mind and the whole
architecture becomes easy to reason about — see file 03.

---

## 1.5 Masks: the whole reason generation works

A mask is added to the logits before the softmax, with forbidden entries set to
`-inf` so that `exp(-inf) = 0` exactly. (In practice one uses a large negative
float, or a fused kernel's mask argument. It must be applied *pre*-softmax; a
"zero out the weights afterwards then renormalise" version is equivalent but
wasteful, and a version that forgets to renormalise leaves rows that do not sum
to 1 — a silent, hard-to-find bug.)

**Causal / autoregressive mask.** `mask[i][j] = (j ≤ i)`. Query `i` sees keys
`0..i`. This is *the* reason a language model can be trained on an entire
document in parallel while still being usable for generation: position `i`'s
output is a function of `x_0..x_i` only, so all `n` predictions are computed
simultaneously and none of them can see the answer.

Lab 5 verifies this properly. A change at position 6 leaves outputs `0..5`
**bit-identical**, while all outputs from position 6 onward move. Remove the
mask and *every* position moves, including positions before the edit — which is
what a bidirectional encoder is supposed to do.

**Padding mask.** In a batch, sequences have different lengths; positions past
the end are masked out for every query. Combined with causality by AND-ing the
two masks.

**Cross-attention mask.** In an encoder-decoder, the decoder queries attend to
all encoder positions (no causal mask on the key axis) — but padding is masked.

**Sliding-window / local mask.** `mask[i][j] = (i - window < j ≤ i)`. Linear
cost. Discussed in file 04.

**The `-inf` hazard.** If an entire row is masked, the row is `-inf` everywhere,
`softmax` computes `0/0`, and you get a NaN which then poisons every downstream
gradient. This is a real bug class, not a curiosity: it happens with fully
padded sequences, with rows whose visible window was entirely evicted, and with
sloppy document masks. Lab 5 demonstrates it. The standard defences are (a)
guarantee at least one visible key per row, e.g. by never masking position 0, or
(b) detect the row and write zeros.

---

## 1.6 Position: attention is a set operation

Attention as defined above has **no notion of order**, and this is not a
limitation you can hand-wave. It is a theorem about the function.

Formally, attention is equivariant to permutation of the input positions: for
any permutation `π`, `Attn(πX) = π Attn(X)`. Proof, in one line: the score
matrix is `(πQ)(πK)ᵀ`, which is `π (QKᵀ) πᵀ`; softmax is row-wise so it commutes
with the row permutation; hence `A' = π A πᵀ`, and `A' (πV) = π (A V)`. Note
that the *attention matrix itself changes* — row 0 attends to a different token
— but the set of output vectors is identical. Lab 6 checks both halves.

The consequence: for a bare attention layer, "the cat chased the dog" and "the
dog chased the cat" are literally the same computation, and a bag of words is
all it can see. Try to get the model to distinguish them and it cannot, at any
scale, with any amount of data. Position must be injected from outside:

| scheme | where it enters | properties |
|---|---|---|
| **Sinusoidal PE** (Vaswani 2017) | added to embeddings | no parameters; `PE[p]·PE[q]` is a function of `p-q`; fixed, extrapolates poorly in practice |
| **Learned absolute** (GPT-2, BERT) | added to embeddings | a `(max_len, d_model)` table; cannot extrapolate past `max_len` at all |
| **RoPE** (Su 2021) | rotates `q` and `k` | `⟨rope(q,m), rope(k,n)⟩` depends only on `m-n`; used by essentially every open model today |
| **ALiBi** (Press 2021) | additive bias on logits | `-slope_h·|i-j|`; zero parameters; extrapolates to unseen lengths |
| **NoPE** | nowhere | surprisingly viable in causal decoders: from the causal mask alone, a head can infer "how far back" from the *shape* of what it can see |
| **Relative bias buckets** (T5) | additive bias on logits | learned per (head, distance-bucket) |

**RoPE deserves the extra paragraph** because it is the current default, and
because its one exact property is easy to state and easy to get wrong:

Each pair of channels `(x_{2i}, x_{2i+1})` in the head dimension is treated as a
2-D point and rotated by `θ_i · position`, where `θ_i = base^{-2i/d_h}`. Applied
to `q` and `k` — never `v` — *after* the projections. Since each pair's rotation
is an orthogonal 2×2 matrix `R(θ_i p)`,

```
⟨R(θ_i m) q_i , R(θ_i n) k_i⟩ = ⟨q_i , R(θ_i (n-m)) k_i⟩
```

so summing over pairs, the score depends only on `n-m`. The frequencies form a
geometric ladder from `θ_0 = 1` (one radian per token, a full turn every
`2π ≈ 6.3` tokens — the finest-grained position signal) down to
`θ_{d_h/2-1} = base^{-1}` ≈ `1e-4` (a full turn every ~63k tokens — the coarsest)
— a multi-resolution clock. Lab 6 and `verify.py` check the relative property at
offsets from 3 up to 2²⁰, plus the two corollaries worth internalising:

1. **The norm is preserved** (a rotation is orthogonal), so RoPE does not change
   the logit scale, i.e. it does not interact with the `1/√d_k` story.
2. **RoPE carries no absolute position information at all.** The score at a
   given offset is *exactly* the same at position 3 and at position 10⁶. Every
   positional signal in a RoPE model is a function of the offset, read by
   learned weights. This is why long-context extension for RoPE models is about
   *interpolating or rescaling the frequencies* rather than about adding more
   positional information — see file 04, §4.7.

---

## 1.7 What attention can and cannot do

The convex-combination property is a hard ceiling, and it is worth being
concrete about what it rules out.

**It cannot compute a function of the *set* that output elements cannot.** Every
output row is in the convex hull of the `V` rows. So attention alone can
implement selection (one-hot `A`), averaging (uniform `A`), and interpolation —
but not "take the first element", not "count the elements" (there is no
counting mechanism: any convex combination of `m` identical values is that
value, regardless of `m`), and not "multiply two values together".

**Counting, sorting, and arithmetic come from the projections and MLPs, not from
the attention pattern.** When you read that a transformer "can implement X", the
typical proof decomposes into (a) attention as a routing/selection mechanism,
(b) the MLP as a per-position function. A single attention layer with
content-independent weights computes a *linear* function of `V`, i.e. is no more
expressive than a matrix; all its power comes from the fact that the weights
depend on the content via `softmax(QKᵀ)`.

**Where the power actually lives.** A single head is one softmax distribution
per query. The empirical and theoretical literature on what that buys:

- *Sparse averaging* (`Sanford, Hsu, Telgarsky 2023`, "Representational
  Strengths and Limitations of Transformers"): a single attention layer solves
  sparse averaging — "return the average of the `k` entries matching this
  query" — with complexity independent of sequence length, where recurrent and
  feedforward networks need complexity growing with `n`. That is the crispest
  formal statement of what attention buys: length-independent access to
  selected content.
- The same line of work gives matching limitations: hard attention-like
  problems need either many heads or multiple layers; a fixed-depth transformer
  cannot solve arbitrary "search" tasks whose answer requires unbounded
  sequential composition.
- Empirically, attention heads implement recognisable, *named* circuits:
  previous-token heads, induction heads (the in-context learning circuit),
  syntactic dependency heads, positional heads, and "sink" heads that do
  nothing. We reproduce two of these on a toy problem in
  `code/tiny_transformer.py` — a previous-token head that reaches 100%
  accuracy, and the query-position attention map of an induction model.

Empirical caveat to keep in mind whenever a paper says "head X does Y": heads
are often redundant (pruning many heads costs almost nothing at inference —
Michel et al. 2019, "Are Sixteen Heads Really Better than One?"; Voita et al.
2019), and identified roles are usually sufficient rather than necessary for
the model's behaviour.

---

Continue to [`02-multi-head.md`](02-multi-head.md) — the same mechanism, run `H`
times in parallel.