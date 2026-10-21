# 03 — Attention inside a real transformer

Lab 3 (shapes), lab 8 (training vs generation), lab 10 (cost).
`code/tiny_transformer.py` is a complete, trainable decoder-only transformer in
~120 lines — read it alongside this file.

---

## 3.1 The block

A modern decoder-only transformer layer is this, in this order:

```python
x = x + Attn(LayerNorm(x))     # communication: tokens exchange information
x = x + MLP(LayerNorm(x))      # computation: each position is transformed
```

Two structural choices hide in that snippet, and both are load-bearing.

### The residual stream

Everything is `x + f(x)`. The model never overwrites its state; it only adds to
it. Consequences:

- **Gradient flow.** The identity path gives every layer a gradient route to
  the loss with a Jacobian of 1, so a 100-layer model does not suffer the
  vanishing-gradient problem a plain deep stack would.
- **The state is a bus, not a buffer.** Any layer may read (`W_q, W_k, W_v`,
  MLP in-projections) or write (`W_o`, MLP down-projection) any direction in
  `d_model` dimensions. This is why interpretability work treats attention as
  *communication* and the MLP as *computation*, and why "the information" is
  never localised in one layer.
- **Attention output and MLP output are just additions.** Ordering within a
  layer only matters through the normalisation placement.

### Pre-norm vs post-norm

The original 2017 transformer was post-norm: `LayerNorm(x + Attn(x))`. That puts
the normalisation *inside* the residual path, so the identity route to the loss
passes through a LayerNorm Jacobian at every layer. Post-norm transformers need
learning-rate warmup and are unstable past ~20 layers.

Pre-norm (`x + Attn(LN(x))`, as above) keeps the residual path a clean identity
and is what essentially every model since GPT-2 uses. The cost is a different
one: the residual stream's variance grows with depth, because each addition adds
variance and nothing renormalises it, so models add a final `LayerNorm` before
the unembedding (and often normalise `q`, `k`, or the logits).

### Where attention sits in the parameter budget

| component | parameters | notes |
|---|---|---|
| attention | `4 d²` | `4 d²` for GQA with `G` groups: `2d² + 2d²·G/H` |
| MLP | `≈ 8 d²` | `d → 4d → d` (SwiGLU: `d → ~2.67d → d`, ≈ `8 d²`) |
| LayerNorms | `4d` | negligible |

So a layer is about one-third attention by parameters — but for `n < 2 d_model`
it is *far* less than one-third of the FLOPs (§3.6). This asymmetry — where the
weights are versus where the compute is — explains a lot of the design of real
models.

---

## 3.2 Attention is the only place tokens interact

Look back at the shape table in §1.4. Everything in the block other than the
`(n, n)` matmuls operates independently per position:

- `q_i, k_i, v_i` — a function of `x_i` alone.
- the MLP — a function of `x_i` alone (`MLP` is applied to the same weights at
  every position; it is *not* a convolution, it has no kernel).
- LayerNorm — a function of `x_i` alone.
- the residual addition — per position.

Only `softmax(QKᵀ) V` reads from more than one position. This is worth stating
as a design principle, because it makes the architecture easy to reason about:

> **Attention is the model's only channel for tokens to influence each other.
> Depth, pure and simple, is what allows information to travel.**

A token at position 0 can influence position 60 directly, in one layer. A token
that must travel through content — "A, which means B, which is caused by C" —
takes layers, and the number of layers, not the context length, is what bounds
reasoning depth. This is why chain-of-thought works: it converts internal
computation into tokens, and tokens get their own fresh layers' worth of
processing at each step.

The "MLP is per-position" fact is also why the MLP can be thought of as a set of
key–value memories: `W_in` rows are patterns, `W_out` rows are what they write,
and the nonlinearity decides which memories fire at each position.

---

## 3.3 Three attention topologies

| | queries | keys | mask | example |
|---|---|---|---|---|
| **Encoder** (bidirectional) | all `n` | all `n` | padding only | BERT, T5 encoder, ViT |
| **Decoder** (causal) | all `n` | all `n` | causal + padding | GPT, Llama, every modern LLM |
| **Cross-attention** | decoder `n` | encoder `m` | padding only | translation models, modern multimodal / audio decoders |

**Why decoder-only won.** Encoder-decoder has two sets of parameters and two
architectures to tune; the decoder-only model does everything with one stack,
because "the encoder's job" can be expressed as a prefix of the same causal
sequence. Cross-attention survives in cases where the two sequences are
genuinely different modalities (image embeddings into text), because then the
`q` and `k, v` really do come from different spaces.

**Prefix-LM / bidirectional masks.** Some models mask so that the prompt is
fully visible to itself while the completion is causal (PaLM, some T5 variants).
It is just a different mask; nothing else changes.

**Bidirectional attention costs no extra machinery but does cost the ability to
train on generated tokens.** The whole reason causal masking is universal in
LLMs is that it lets one forward pass provide a supervision signal at *every*
position simultaneously — you feed a document once and get `n` next-token
predictions. Lab 5 shows the property that makes this valid: a change at
position `i` provably cannot affect the output at any `j < i`.

---

## 3.4 Training: every position at once

The teacher-forced training step is:

```
input   x_0 .. x_{n-1}
targets x_1 .. x_n           (the same sequence shifted by one)
loss    cross-entropy at every position, averaged
```

One forward pass, `n` predictions, `n` gradients. The difference between the
prediction at position `i` and the target is what teaches the model
`P(x_{i+1} | x_0..x_i)`.

**Important subtlety, easily missed.** What the model is trained to do is
predict the next token *given a document that already exists*. What it does at
inference is *continue a document it is writing*. The distribution of prefixes
during self-generated decoding drifts away from the training distribution
(exposure bias) — which is why a pretrained model can look excellent on
validation loss and degenerate into repetition in a multi-turn or long-form
setting. It is also why RLHF/DPO exist at all: they operate on the model's own
outputs, not on a teacher-forced corpus.

**One more thing worth internalising:** with a causal mask, position `i`'s
attention pattern is computed over a *variable-length* key set (`i+1` keys out
of `n`). Early positions have very little to attend to. This asymmetry is why
"the first token" is special all through the stack (sinks, §4.6), and why
masked-out positions must be handled explicitly rather than assumed away.

---

## 3.5 Inference: two modes, and the KV cache

Generation is sequential. You have `x_0..x_t`, you compute the logits for
position `t+1`, sample, append, repeat.

The naive implementation recomputes the entire forward pass for the whole
sequence at every step: `O(n²)` work to generate `n` tokens, with total cost
`Σ_t O(t²)`... but you don't need to. Because attention is causal, the
**attention output at position `t` never depends on positions after `t`**. So
the `K` and `V` projected for positions `0..t-1` at step `t` are the same values
that will be needed at step `t+1`. Store them:

```
step t:  x_t  ->  q_t, k_t, v_t                       # one token's worth
         append k_t, v_t to the cache                 # O(1) memory write
         A_t = softmax(q_t K_cacheᵀ / √d_h)           # attend over everything
         out_t = A_t V_cache
```

That is the **KV cache**, and it is the single most important implementation
detail in inference engineering. Two consequences:

- **Work per token becomes `O(n·d)`** instead of `O(n²·d)` recomputation, and
  the total for generating `n` tokens is `O(n²·d)` — which is, note, the same
  total as one parallel forward pass. You cannot beat `O(n²)` for a full
  sequence of length `n`; you can only avoid doing it `n` times.
- **Memory becomes the binding constraint.** The cache is
  `2 · L · H_kv · d_h · ctx · 2` bytes in fp16, which is 2 GiB for Llama-2-7B at
  4096 context and 16 GiB for Llama-3-8B at 128k — *per sequence in the batch*.
  File 02 §2.5 has the table; file 04 §4.4 has the architectures designed
  around it.

### The cache is exact, not an approximation

`verify.py::t_kv_cache_equals_full_forward` and lab 8 check it: stepping token
by token through a cache reproduces the parallel forward pass to ~1e-15. This is
the fact that makes the whole picture consistent — **the same weights serve both
modes**, and the only difference is that one of them materialises `n` queries at
once while the other materialises one.

### Prefill and decode are different workloads

| | prefill | decode |
|---|---|---|
| what | process the prompt, all `n` tokens at once | one token at a time |
| bottleneck | compute (a big, dense, square matmul) | memory bandwidth (reading the cache and the weights) |
| scalable by | better kernels, tensor parallelism | bigger batches, KV compression, quantization |
| what it looks like | a few large GEMMs | many tiny GEMV-ish operations, latency-bound |

Every real serving system (vLLM, TensorRT-LLM, SGLang) is organised around this
split: *chunked prefill*, *continuous batching*, *paged attention* (fixed-size
cache blocks with a page table, to kill fragmentation), *prefix caching*
(reusing the cache for a shared prompt prefix — which works precisely because
attention results for a prefix are independent of what follows it).

---

## 3.6 Cost, and where attention stops being cheap

Per layer, per sequence, forward pass, forward-only FLOPs:

```
QKᵀ : 2 · n · m · d          AV : 2 · n · m · d          -> attention ≈ 4 n m d  (m = n for self-attn)
up   : 2 · n · d · 4d        down: 2 · n · 4d · d        -> MLP         ≈ 16 n d²
```

Setting them equal:

```
4 n² d = 16 n d²      =>      n = 4 d
```

So the crossover — the sequence length at which attention starts to dominate
the FLOPs — is a few thousand tokens for a `d = 4096` model, well within the
context windows real models use. **Below** that, the model is mostly an MLP
network and attention is a rounding error; **above** it, the model is mostly an
attention machine and the MLP is the cheap part. Lab 10 measures the quadratic
scaling directly (doubling `n` quadruples the time and the memory).

Two practical corollaries:

- Long-context research is almost entirely attention research.
- Attention has **no parameters in the mechanism itself**. The `(n, n)` matmul
  is a fixed routing function over whatever the four projections produce. All
  the learning lives in `W_q, W_k, W_v, W_o` — which is why a "pruned" or
  "distilled" attention pattern can be borrowed from a big model at no
  parameter cost.

---

## 3.7 How the pieces fit into a whole model

A concrete, complete stack (Llama-3-8B), so you can see the proportion:

| | value |
|---|---|
| layers | 32 |
| `d_model` | 4096 |
| query heads | 32 (`d_h = 128`) |
| KV heads (GQA) | 8 |
| MLP intermediate | 14336 (SwiGLU) |
| normalisation | RMSNorm, pre-norm |
| positional | RoPE, base 500000 |
| attention impl | causal; flash-attention kernel in practice |
| vocab | 128256 |
| context | 8192 (trained), extended later |

Read that list as the answer to "what does the attention part of a modern model
actually look like": GQA, RoPE with a large base, pre-norm, a fused kernel, no
bias terms, no `softmax1` — the boring, robust version of everything in these
notes.

---

Continue to [`04-modern.md`](04-modern.md) — efficiency, variants, and what
current models actually do.