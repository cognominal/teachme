# 04 — Efficiency, variants, and what real models do

Labs 9, 10, 11, 12, 13 in `code/lab.py`. Numbers below that are specific to
released models are labelled with the source; everything else is derived in the
labs and reproducible.

---

## 4.0 The three costs, and which one binds

Attention's cost is not one thing:

| resource | scaling | where it binds |
|---|---|---|
| **FLOPs** | `O(n² d)` | prefill of long sequences; training |
| **activation memory** | `O(n²)` for the score matrix; `O(n·d)` for the projections | training activations; the score matrix is a *transient* |
| **bandwidth / cache memory** | `O(n·d·L)` for the KV cache | *every* decode step, always |

Lab 13 puts a number on the second row: at `n = 2048, d = 4096, H = 32`, the
score matrix is `2048 × 2048 × 32 × 2 bytes = 256 MiB` — **16× the input
activation it was computed from, and twice the size of the layer's entire set
of attention weight matrices** (`4 d² × 2 bytes = 128 MiB`). And it is written
and read once per layer. Meanwhile the operands it was computed from are a few
MiB.

An A100 does ~19.5 TFLOP/s fp16 but has ~2 TB/s of HBM bandwidth, i.e. roughly
**100 FLOPs of budget per byte moved**. If a kernel does fewer than ~100 FLOPs
per byte it touched, it is bandwidth-bound and its FLOP rating is fiction.
Attention is squarely in that regime. This is the single most useful lens for
reading efficiency work:

> Count bytes moved first, FLOPs second. An optimisation that reduces arithmetic
> but not traffic is usually worthless; one that reduces traffic wins even at
> identical arithmetic. Flash attention is exactly that: same FLOPs, ~10× fewer
> bytes.

The roofline intuition also explains why the KV cache dominates production: it
is *pure bytes*, linear in context, re-read on every single decode step, and
there is no arithmetic to hide it behind.

---

## 4.1 The complexity, stated precisely

Per layer, forward pass:

```
QKᵀ           2 n² d        score matrix, materialised in the naive version
softmax       ~5 n²          cheap relative to the matmuls
AV            2 n² d
--------------------------------------------------------------
attention     ≈ 4 n² d
MLP           ≈ 16 n d²      2nd·4d up + 2n·4d·d down
```

Lab 10 measures the quadratic scaling on a real machine: doubling `n` roughly
quadruples the time and the memory. The crossover with the MLP is at
`4 n² d = 16 n d²`, i.e. **`n = 4 d_model`** (file 03, §3.6) — a few thousand
tokens for a real model.

Three structural facts that follow, and that shaped everything below:

- **`O(n²)` is not avoidable in general.** Softmax attention must, in the worst
  case, look at every key for every query: the log-sum-exp denominator couples
  all keys together. You can avoid *materialising* it (flash, §4.2), avoid
  *reading all of it* (sparse, §4.3), or change the function so the coupling
  factorises (linear attention, §4.5). You cannot make exact softmax attention
  sub-quadratic without an approximation.
- **The `(n, n)` tensor is transient.** It is consumed immediately to compute
  `AV`. That means it never *has* to exist in memory — which is the entire idea
  of flash attention.
- **Attention parameters are only `4 d²`.** Everything expensive about attention
  is sequence-length-dependent, not width-dependent.

---

## 4.2 Flash attention: online softmax

The naive kernel computes `A = softmax(QKᵀ)` and then `AV`. That means writing
an `(n, n)` tensor to HBM, reading it back, writing the softmaxed version,
reading it back. Three round trips for a tensor that is orders of magnitude
bigger than the operands.

But the softmax needs a row *maximum* and a row *sum* before it can scale
anything — and those need the whole row. That looks like it forces the whole row
into memory. It does not:

```
for each block of keys:
    S = Q_block K_blockᵀ / √d
    m_new = max(m_old, rowmax(S))                  # new running maximum
    α     = exp(m_old - m_new)                     # rescale factor
    l     = l * α + rowsum(exp(S - m_new))         # running denominator
    acc   = acc * α + exp(S - m_new) V_block       # running numerator
output = acc / l
```

The algebra is one line, and it is exact:

```
exp(s_j - m_new) = exp(s_j - m_old) · exp(m_old - m_new)
```

so multiplying the accumulated numerator `acc` and the denominator `l` by the
same factor `α` leaves `acc/l` — the attention output — completely unchanged. You
never need to see the whole row before you can start accumulating. `acc/l` is
the correct softmax-weighted sum at every intermediate stage; if you stopped
early you would have the answer for the keys seen so far.

This is **online softmax**, and flash attention (Dao et al. 2022) is that
identity plus two systems tricks:

1. **Tiling** — process `Q` in blocks of ~128 rows and `K, V` in blocks of ~128
   columns, so the score block (`128 × 128`) and the accumulators stay in
   registers / shared memory. Never touch HBM for the score matrix.
2. **Causal skipping** — with a causal mask, whole key blocks that lie entirely
   in the future are skipped, so you do half the work. `online_attention` in
   `attn.py` implements this (`if causal and j0 >= i1: break`).

Backward passes are handled by *recomputing* the score block on the fly rather
than storing it (or by storing just the per-block log-sum-exp, `O(n)` instead of
`O(n²)`). That is the memory-for-compute trade in the opposite direction from
usual, and it wins because the recompute is cheap and the bytes are not.

Lab 9 checks it, numerically, on the real implementation in `attn.py`:

```
n = 2048, d_head = 64
naive  :  65.5 ms,  score matrix  32.00 MiB allocated
blocked:  28.3 ms,  largest working set  128.0 KiB
max |naive - blocked| = 6.661e-16      <- exact, not approximate
```

**You must internalise that this is not an approximation.** Same function, exact
same output, different memory access pattern. `verify.py` checks it against the
naive version across block sizes including non-divisible ones (block 7 on
`n = 100`) and with a padding mask mixed into the causal mask.

In practice you do not write this kernel: `torch.nn.functional.
scaled_dot_product_attention` dispatches to a flash, memory-efficient, or
math kernel depending on dtype, mask type, and device — and the *math* kernel
is what `verify.py` compares against, so the code here is checked both ways.

---

## 4.3 Sparse and local attention

If the `(n, n)` matrix is mostly irrelevant, do not compute it.

**Sliding window (local) attention.** `mask[i][j] = (i - window < j ≤ i)`.
Cost `O(n · w · d)` — linear in `n`. Lab 11 prints the mask and the cost:

```
at n=1000, window=64:  98,059 of 1,000,000 pairs (9.81%)
```

The obvious objection — "doesn't this break long-range dependencies?" — has a
good answer: **information still travels further than `w` tokens, by hopping
through layers.** Token `i` can tell token `i+w` which can tell `i+2w`, so after
`k` layers the receptive field is `k·w`, while the cost stays `O(n·w·layers)`.
Mistral-7B is a pure sliding-window model with `w = 4096`. Whether the *learned
computation* can actually be decomposed into hops is an empirical question, and
the answer has been "largely yes, with some loss on tasks requiring exact
long-range retrieval".

**Alternating local/global layers.** The current dominant pattern: mostly cheap
local layers, with a full-attention layer every `k`-th layer to handle the
long-range case. Verified from the released Gemma-3 configuration
(`config.json`, Gemma 3 12B, read directly):

```
head_dim                 256        hidden_size          3840
num_attention_heads       16        num_key_value_heads     8   (GQA, group 2)
num_hidden_layers         48        intermediate_size   15360
sliding_window          1024        sliding_window_pattern   6   (5 local : 1 global)
rope_theta           1000000.0      rope_local_base_freq  10000.0
rope_scaling   {"rope_type": "linear", "factor": 8.0}
max_position_embeddings 131072      attn_logit_softcapping  null
```

Three details there are worth more than the headline number:

- `sliding_window_pattern: 6` means one global layer per six, so **only 8 of 48
  layers** keep a full-length KV cache. The cache is not `48 × n`, it is
  `8 × n + 40 × min(n, 1024)`.
- **Local and global layers use different RoPE bases** (10⁴ vs 10⁶). The local
  layers do not need the low-frequency components, and giving them a smaller
  base improves resolution at short range. Two positional regimes in one model.
- `rope_scaling: linear, factor 8` with `rope_theta 1e6`: 128k context from
  16k training length by shrinking the effective rotation.

**Block-sparse and structured sparsity.** Longformer/BigBird-style patterns:
local blocks + a few global tokens + strided connection. Approximates attention
at `O(n√n)` or `O(n)`. Largely superseded — sliding-window-plus-alternating is
simpler and trains better.

**What actually matters in practice:** sparse patterns change the attention
*function*, so the model must be trained with them (or distilled into them from a
dense model — exactly what "sparse attention" papers do). They are not a
drop-in inference optimisation on a dense model. The drop-in optimisations are
in §4.4 and §4.8, and they are the ones deployment engineers actually use.

---

## 4.4 Cache compression: the architectures

Since the KV cache is `O(n · L · H_kv · d_h)` bytes and binds at every decode
step, shrinking `H_kv` or `d_h` is the highest-leverage change available.
File 02 §2.5 covers MQA/GQA; the reversal to know is **why GQA is "free"**:

- The KV projection shrinks (`W_k`, `W_v` become `d × G·d_h` instead of
  `d × H·d_h`), so the model loses parameter capacity in the key/value *content*
  — but the query side is untouched, so a GQA model still computes `H`
  independent attention distributions.
- Empirically the information loss is tiny for `G` in the 4–16 range; a trained
  GQA model at 8 KV heads is very close to MHA. This is why Llama-2-70B, Llama-3
  and almost everything after uses GQA, while Llama-2-7B (which has a small
  enough cache) does not.

**MLA — Multi-head Latent Attention** (DeepSeek-V2, DeepSeek-V3). A different
move: instead of having fewer KV heads, cache a *compressed latent* and
re-expand K and V on the fly.

- Per layer, per token, the cache holds a latent `c_KV` of dimension
  `kv_lora_rank = 512` plus a decoupled RoPE part of 64, i.e.
  **576 numbers**, in place of `2 × H × d_h`.
- At `H = 128, d_h = 128` that would be `2 × 128 × 128 = 32768` numbers —
  a ~57× reduction in raw element count, and DeepSeek report a 93.3% KV-cache
  reduction relative to their dense-attention baseline.
- The catch, and the clever part: the up-projections `W_UK, W_UV` map the latent
  back to per-head K and V. If you naively expand on every decode step you pay
  the re-expansion cost. The **absorb trick** folds those matrices into `W_q`
  and `W_o` instead, so the expander never has to be materialised at inference.
  The cache never grows back. Separately, a small per-head RoPE component is
  kept un-merged, because RoPE cannot be folded (it is position-dependent), so
  the cache is `c_KV` + `k_rope` and the model is exact, not an approximation.

**Quantisation.** The KV cache is the cheapest tensor in the model to quantise:
it is written once, read many times, and small errors in K and V compound far
less badly than errors in the weights. int8 KV cache is standard; fp8 is common
on Hopper/Blackwell. Note the failure mode: **outlier channels in K** dominate
the quantisation error because the dot product is sensitive to a few large
coordinates. Per-channel (per-head-dim) scales fix most of it.

**Eviction and compression over time.** Keep only some tokens' KV:

- *StreamingLLM* (Xiao et al. 2023): keep the first few tokens plus a sliding
  window. It works because of the attention sink (§4.6) — the first tokens'
  KV entries are receiving huge attention mass and cannot be dropped.
- *H2O*-style heavy hitters: keep the tokens with the highest accumulated
  attention mass. Highly lossy; can be catastrophic on tasks where the needed
  detail was never salient early.
- *Learnable compression*: cache compression modules that store a fixed budget
  of KV slots and train the model to use them (e.g. ICAE, "gist" tokens,
  Landmark Attention).

**Serving-layer tricks** (not model changes, but they matter more in practice
than most model changes): *paged attention* stores the cache in fixed-size
blocks with an indirection table, eliminating fragmentation and allowing
prefix sharing; *prefix caching* reuses the cache for a common prompt prefix —
valid precisely because a prefix's attention results do not depend on what
follows (file 03, §3.5); *chunked prefill* interleaves prefill and decode steps
so a long prompt does not stall the batch.

---

## 4.5 Linear attention, and what it gives up

The softmax is what couples all keys together. Replace `softmax(qᵀk)` with a
kernel factorisation `φ(q)ᵀ φ(k)` and the sum rearranges:

```
out_i = Σ_j  φ(q_i)ᵀφ(k_j) v_j / Σ_j φ(q_i)ᵀφ(k_j)
      = φ(q_i)ᵀ (Σ_j φ(k_j) v_jᵀ) / φ(q_i)ᵀ (Σ_j φ(k_j))
```

The bracketed terms are a fixed-size **state** (`d_φ × d_v` and `d_φ`), updated
as a running sum. Cost per token: `O(d_φ · d_v)`, total `O(n · d_φ · d_v)` —
**linear in `n`, and no KV cache at all**, because the state is what you keep.

That sounds like a free lunch. It is not, and the reason is the most instructive
thing in this file:

- Softmax attention is a **sharp** lookup. Its weights are exponential in the
  score, so it can pick out one key out of a million and ignore the rest
  (up to the sink caveat). A kernel factorisation with a polynomial or
  exponential-with-bounded-range feature map **cannot** represent that sharp
  peak, because the state it accumulates is a fixed-size average-like object.
  Once you have summed over a million keys into a fixed-size state, retrieving
  one of them is not possible.
- Hence the empirical pattern: linear attention is competitive on many language
  modelling metrics and loses specifically on **recall / copying / associative
  retrieval**, i.e. exactly the induction-head behaviour from file 02.
- Hence the current state of the art in practice: **hybrids**. Keep a small
  number of full-attention layers for the retrieval work and make the rest
  linear, or interleave attention blocks with state-space blocks
  (Jamba, Zamba, Samba, Griffin, and the various "attention + Mamba" mixes).

**RWKV and Mamba are not attention.** They are recurrences with fixed-size
state: `O(n)` and no cache, but they inherit the recurrent bottleneck from file
01 §0 — everything must be compressed into a fixed-size state, and training is
parallelised only by exploiting the linear-recurrence structure (a scan). Their
"attention-like" behaviour comes from *selective* gating, which is a mechanism,
not a softmax over a growing key set. Reading a paper that calls them "attention
alternatives" is fine; reading one that calls them attention is not.

The honest summary: **softmax attention's `O(n²)` buys exact, unbounded,
content-addressed retrieval.** Every cheaper alternative trades that away
partially, and the empirical work since 2020 has consistently found that
retrieval is worth paying for on a few layers.

---

## 4.6 Softmax pathologies, and the fixes people have shipped

### The attention sink

Recap from file 01 §1.2: a softmax row must sum to 1, so a head with nothing to
look at must still put its mass somewhere. Trained models overwhelmingly dump it
on the first token. That is the *attention sink* (Xiao et al. 2023, whose
StreamingLLM paper documents it and argues the mechanism is precisely the
normalisation constraint), and lab 12 makes the pressure visible without a
model: with 1000 irrelevant keys at logit std 4, the *worst* key still receives
`1e-3` of the mass, and the top key takes the lion's share of a quantity that
cannot be shrunk.

Consequences you can observe directly in a real model's attention maps:

- a large fraction of heads put more than half their mass on token 0, and those
  tokens typically carry no semantic content;
- evicting the first tokens' KV entries destroys quality for sliding-window
  inference — hence StreamingLLM keeps them permanently;
- sink behaviour appears in vision and audio transformers too, so it is a
  property of softmax, not of language.

### Fixes that have been proposed for it

| approach | idea | status |
|---|---|---|
| **softmax1 / "quiet attention"** (Miller 2023) | `exp(z_j)/(1 + Σ exp(z_j))` — allow the total to be < 1, i.e. "attend to nothing" | elegant; ~0.5 lines of code; small reported quality gains; requires retraining |
| **sigmoid attention** | replace `softmax` with `σ(·)` per logit, drop normalisation | works in vision models; unstable in LLMs without extra normalisation |
| **Differential attention** (Ye et al. 2024) | compute two attention maps, subtract them — the sink is common to both and cancels | reported gains; ≈2× the attention compute |
| **always-present sink token** | add a null token to the vocabulary and let the head attend to it | explicit, cheap, robust |
| **keep the first tokens** (StreamingLLM) | do not evict them | a deployment fix, not an architecture fix |

### Logit-scale pathologies

Same family: logits that get too large saturate the softmax (file 01 §1.3).

- **QK-norm** normalises `q` and `k` before the dot product, bounding the logit
  range structurally. Used in ViT-22B, Gemma 2/3 (via `query_pre_attn_scalar`
  and per-layer normalisation), and many 2024+ models.
- **Logit soft-capping** pushes the logits through `softcap · tanh(z/softcap)`,
  bounding them by construction. Gemma 2 used it (`attn_logit_softcapping: 50`,
  `final_logit_softcapping: 30`); the released Gemma 3 configuration has both
  set to `null` (verified above) — an example of the field converging on the
  simpler fix.

---

## 4.7 Long context

Two separate problems hide under one name, and conflating them is the most
common mistake in this area.

**Problem 1: the positional encoding misbehaves past its training length.**
RoPE frequencies are geometric, so at positions far beyond training, the fast
frequencies alias (a token at position `p` and `p + 2π/θ_0` are indistinguishable
to the highest-frequency pair) and the model meets relative offsets it never
trained on. Fixes, all of which alter frequencies or positions rather than the
attention mechanism:

- **Position interpolation** — scale positions down so the model sees only
  offsets it trained on (`rope_scaling: linear, factor 8` in the Gemma 3 config
  above is exactly this). Cheap, loses resolution at short range.
- **NTK-aware scaling** — scale the base rather than the positions, non-uniformly
  across frequencies: fast frequencies barely change (short-range resolution
  preserved), slow ones stretch a lot.
- **YaRN** — the above, plus a per-frequency ramp between "interpolate" and "do
  nothing", plus an attention-temperature correction, and a small fine-tune.
- **A bigger base from the start** — Llama-3 raised `rope_theta` from 10⁴ to
  **5·10⁵** and trained at 8k, then extended; Gemma 3 uses **10⁶** for its global
  layers. (Both numbers read from released configs.)
- **LongRoPE / dynamic scaling** — per-dimension or per-step rescaling, for
  100k+ contexts.

**Problem 2: the model was never trained on long dependencies.** Extending the
positional encoding gives you a mechanically valid 128k forward pass. It does
not give you a model that *uses* 128k. Attention mass spreads thin as context
grows (softmax over more equally-plausible keys), and there is no signal in
pretraining at 4k that teaches long-range retrieval. This is why
needle-in-a-haystack scores look great while real long-document reasoning
degrades, and why every serious long-context model does a long-context
*training* stage, not just a positional rescaling.

**The cheap version of long context** is a retrieval system, and it beats naive
long context on cost and often on quality. Worth knowing as the baseline your
128k attention has to beat.

---

## 4.8 What to look at in a config file

Given the above, you can read any 2024–2026 attention config in about a minute.
The Gemma 3 block in §4.3 is a good template. The questions, in order of how
much they change the cost:

1. `num_key_value_heads` vs `num_attention_heads` — MHA, GQA, or MQA? This sets
   the KV cache, i.e. the dominant inference cost.
2. `sliding_window` / `sliding_window_pattern` or an alternating-attention
   flag — how many layers keep a full-length cache?
3. `head_dim` and `num_attention_heads` — note these need not multiply to
   `hidden_size` (Gemma 3: `16 × 256 = 4096 ≠ 3840`), so do not assume
   `d_h = d_model / H`.
4. `rope_theta`, `rope_scaling` — what positional regime, and what context
   extension was applied?
5. `attn_logit_softcapping`, `query_pre_attn_scalar`, any QK-norm flag — what is
   bounding the logit scale?
6. `attn_implementation` at load time — `sdpa` / `flash_attention_2` / `eager`.
   This changes nothing about the mathematics and a great deal about the speed.

The pattern across 2025–2026 releases: **GQA or MLA, RoPE (usually with a large
base and some rescaling), pre-norm, RMSNorm, SwiGLU, causal, a fused attention
kernel, and an alternating local/global layer pattern for long context.** The
mechanism from file 01 has not changed since 2017. Everything since has been
about the `(n, n)` tensor and the KV cache.

---

## 4.9 The short version, if you remember nothing else

1. Attention is `softmax(QKᵀ/√d_k) V`. One matmul chain. That is the whole
   mechanism, unchanged since 2017.
2. The output is a **convex combination of values** — attention routes, it does
   not compute. Computation lives in the MLPs.
3. **Position is not in the mechanism.** Without RoPE/ALiBi/PE, attention is a
   set operation. With RoPE, scores depend only on the relative offset.
4. **Causality comes from a mask**, and the mask is what makes parallel training
   of a generative model sound.
5. **Multi-head = multiple routers**, combined linearly by `W_o`; the KV side is
   a separate, cheaper parameter that exists to shrink the cache.
6. **The `(n, n)` tensor** and the **KV cache** are the two things every real
   system is organised around. Everything else is a projection.