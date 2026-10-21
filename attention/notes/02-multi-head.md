# 02 — Multi-head attention

Labs 3, 7 in `code/lab.py`. Training evidence in `code/tiny_transformer.py`.

---

## 2.1 The definition

Run `H` independent attention operations in parallel, each in its own
`d_h = d_model / H` subspace, then combine the results with a learned linear
map:

```
head_h     = Attn(x W_q[h], x W_k[h], x W_v[h])       # (n, d_h)
MHA(x)     = concat(head_1, ..., head_H) W_o          # (n, d_model)
```

In code, with `q, k, v` already split into `(B, H, n, d_h)`:

```python
out = softmax(q @ k.transpose(-1, -2) / sqrt(d_h)) @ v   # (B, H, n, d_h)
out = out.transpose(1, 2).reshape(B, n, d_model) @ W_o
```

Three things to notice, because they are all easy to get wrong:

1. **Every head has its own `W_q`, `W_k`, `W_v`.** That is the entire source of
   head diversity: same input `x`, different projections, therefore different
   attention patterns *and* different values read out. A single shared `W_q`
   across heads would make every head's pattern identical.
2. **The softmax and the division by `√d_h` are per head.** The scale uses
   `d_h`, not `d_model`. This is why attention behaves the same whether a model
   has 8 heads of width 128 or 32 heads of width 32.
3. **`W_o` is not just a concatenation.** It is a full `d_model × d_model`
   matrix, so the head outputs are linearly mixed: `W_o` can *add* head `a` to
   head `b`, *subtract* one from another, weight them differently per output
   channel, or cancel a head entirely. This matters — the "heads are
   independent" picture holds for the attention patterns, not for the output.

### Parameter count

Per layer, a head has `d_model · d_h` weights in each of `W_q`, `W_k`, `W_v`,
so `3 · H · d_model · d_h = 3 d_model²`, plus `W_o` at `d_model²`. Total:

```
MHA parameters = 4 d_model²          (plus biases, if any)
```

Compare with the feed-forward block at `≈ 8 d_model²` (two `d_model × 4d_model`
matrices). So a transformer layer is roughly **1/3 attention, 2/3 MLP** by
parameters. Keep that ratio in mind when reading efficiency papers: attention
is not where the weights are, it is where the *quadratic* cost is.

With grouped-query attention (§2.5), `W_k` and `W_v` shrink to
`2 d_model² / group_factor` and the total drops accordingly.

---

## 2.2 Why more than one head

### The algebraic reason

One head produces **one** softmax distribution per query, hence **one** convex
combination of value vectors:

```
out = Σ_j a_j V_j ,     a_j ≥ 0 ,  Σ_j a_j = 1
```

If the answer requires two different positions' information, a single head must
compromise between them. Lab 7 makes this concrete with a case that is not
merely hard, but impossible:

```
values:  V = [[10, 0], [0, 10], [0, 0]]
wanted:  [10, 10]
```

Reaching the first coordinate needs `a_1 = 1`; reaching the second needs
`a_2 = 1`; but `a_1 + a_2 ≤ 1`. The best a single head can do is `[5, 5]` —
Euclidean error `7.07`, searched exhaustively over the simplex in the lab. No
amount of training fixes this, because it is a statement about the *range* of
the function, not about optimisation. Two heads with two independent
distributions reach it; the residual error is `exp(-logit gap)` and the lab
shows it collapsing from `5e-2` to `0` as the logit gap grows from 8 to 64.

That is the honest statement of "why multiple heads": `H` heads give the layer
`H` **independent** routing decisions per position, recombined by `W_o`. A
single head is one router; multi-head is a mixture of routers.

### The theoretical reason

`Sanford, Hsu, Telgarsky 2023` (Representational Strengths and Limitations of
Transformers) analyse sparse averaging, where a layer must return the average of
the `k` sequence entries matching a query. Single-head attention with
content-independent weights can only produce one weighted average, so it scores
at chance; with multiple heads (or extra dimensions), the layer can select and
average, at complexity independent of sequence length — a task where recurrent
and feedforward architectures need complexity growing with `n`.

### The empirical reason

Trained heads specialise, and the specialisations have names. This is the
central object of study in mechanistic interpretability:

- **Previous-token heads** — attend to `i-1`. `code/tiny_transformer.py`
  trains one from scratch: on the "predict the previous token" task the model
  reaches 100% held-out accuracy and one head (layer 0, head 3) puts ~0.99 of
  its weight on exactly position `i-1`.
- **Induction heads** — "find the earlier occurrence of the token I am
  currently on, and copy what followed it". This is the canonical in-context
  learning circuit (Olsson et al. 2022, *In-context Learning and Induction
  Heads*), needs at least two layers, and is what allows a model to continue a
  pattern it has just seen in the prompt. `code/tiny_transformer.py` trains one
  on associative recall, and reports the pointer accuracy of each head.
- **Sink heads** — dump their weight on the first token regardless of content
  (§4.6).
- **Syntactic and positional heads** — attend to the head of a dependency, the
  previous noun, a fixed offset, the end of a clause. Voita et al. 2019 found
  that pruning some heads breaks syntax while others can be removed for free.

The empirical caveat matters as much as the examples: **heads are largely
redundant.** Michel et al. 2019 ("Are Sixteen Heads Really Better than One?")
showed that at *test* time a majority of heads can be pruned with almost no
loss, and that a small number of heads carry most of the importance. So
"head `h` detects X" should be read as *this head is sufficient to carry X*,
not *removing it will break X*.

---

## 2.3 How many heads, and how wide?

The standard configuration is `d_h = 64` or `128`, with the head count derived
as `d_model / d_h`:

| model | `d_model` | heads | `d_h` |
|---|---|---|---|
| GPT-2 small | 768 | 12 | 64 |
| GPT-2 large | 1280 | 20 | 64 |
| GPT-3 | 12288 | 96 | 128 |
| Llama-2-7B | 4096 | 32 | 128 |
| Llama-2-70B | 8192 | 64 | 128 |
| Llama-3-8B | 4096 | 32 | 128 |
| Gemma 3 12B | 3840 | 16 | **256** (`16 × 256 = 4096 ≠ 3840`) |

`d_h` is remarkably stable while `d_model` varies by 16×. The reason is
hardware: a head dim of 64–256 makes the per-head matmuls `(n, d_h) × (d_h, n)`
and `(n, n) × (n, d_h)` the right shape for a fused kernel, and it keeps each
head's register tile small enough to fit on-chip.

Two things that surprise people here:

- **`n_heads × d_h` need not equal `d_model`.** Gemma 3 12B has
  `hidden_size = 3840`, `num_attention_heads = 16`, `head_dim = 256`, i.e.
  `16 × 256 = 4096 ≠ 3840` (read from the released config). `W_q` is
  `(3840, 4096)` and `W_o` is `(4096, 3840)`. So `d_h = d_model / H` is the
  *common* case, not a law.
- **`d_h` is not always small.** Gemma 2 and 3 use 256. Larger head dims give
  each head a richer value space but fewer independent routers, and the
  hardware sweet spot has been drifting upward as kernels improve.

Given `d_h` fixed, "more `d_model`" mechanically means "more heads", which is
why head count is usually reported as a consequence rather than a choice.

The tradeoff if you do vary `d_h` at fixed `d_model`: narrow heads (`d_h = 32`)
give more independent routers but each has a lower-dimensional value space to
read from; wide heads (`d_h = 256`) give each router more to work with but fewer
independent routes. Empirically 64–128 dominates, but the difference is small
and the hardware argument is probably decisive.

---

## 2.4 What a head *is*, in the residual stream

The clean way to see multi-head attention is as `H` independent **rank-limited
writes into the residual stream**.

Head `h` produces context `c_h ∈ R^{d_h}`, and `W_o` maps it back:

```
MHA(x) = Σ_h  c_h W_o[h]           # W_o[h] is the d_h × d_model block-row for head h
```

So each head's contribution is a vector of **rank at most `d_h`** written into
`d_model` dimensions, and the residual stream simply adds them together. This
gives the standard mental model used throughout interpretability work:

```
residual stream  =  a shared, additive communication bus, width d_model
attention        =  communication: moves information *between* positions
MLP              =  computation: transforms information *at* a position
```

Every head reads from the bus (via `W_q, W_k, W_v`), writes back to it (via
`W_o`), and never writes to a head-local buffer. "Head 7 attends to the verb" is
really "head 7's key projection happens to read a direction in the residual
stream that the verb token wrote there". This is why the same head can appear to
do unrelated things on different inputs, and why the "one head, one role" story
is an approximation.

---

## 2.5 Grouped-query attention (GQA) and multi-query attention (MQA)

Everything above describes *query* heads. The number of **key/value** heads is a
separate, free parameter, and it turns out to be the important one at inference
time.

At generation time, the cost that binds is not FLOPs but the KV cache: you must
store `K` and `V` for every past token, for every layer and every KV head (file
03, §3.5). Shrinking the *number of KV heads* shrinks that cache linearly, and
changes nothing about the query side — the model still computes `H` attention
distributions.

| variant | KV heads | KV cache (relative) | how |
|---|---|---|---|
| **MHA** | `H` | 1× | one K and one V per query head |
| **GQA** (Ainslie et al. 2023) | `G < H` | `G/H` | each KV head serves `H/G` query heads |
| **MQA** (Shazeer 2019) | 1 | `1/H` | all query heads share one K and one V |

The implementation is two lines (`attn.py::repeat_kv`): project to `G` heads,
then `np.repeat` each KV head `H/G` times along the head axis so shapes line up.
`verify.py` checks this against `torch.repeat_interleave` and against PyTorch's
`scaled_dot_product_attention`.

Concrete arithmetic for the cache, from lab 8 (`2 · layers · H_kv · d_h · ctx ·
bytes`):

| model | L | H_kv | d_h | ctx | fp16 cache |
|---|---|---|---|---|---|
| GPT-2 small | 12 | 12 | 64 | 1024 | 36 MiB |
| Llama-2-7B | 32 | 32 | 128 | 4096 | 2048 MiB |
| Llama-2-70B (GQA) | 80 | 8 | 128 | 4096 | 1280 MiB |
| *same, if it were MHA* | 80 | 64 | 128 | 4096 | 10240 MiB |
| Llama-3-8B (GQA) | 32 | 8 | 128 | 8192 | 1024 MiB |
| Llama-3-8B @128k | 32 | 8 | 128 | 131072 | 16384 MiB |

A batch of 32 at 128k context for Llama-3-8B needs **512 GiB** of KV cache
alone. That single number is why GQA exists, why Llama-2-70B uses it while
Llama-2-7B does not, why MLA exists (file 04, §4.4), and why almost every
long-context deployment is memory-bound rather than compute-bound.

The quality cost is small — MQA has a noticeable quality gap, GQA at 8 KV heads
is close to MHA — and it trades against parallel-scan efficiency, which is why
GQA (not MQA) is the modern default.

---

## 2.6 Related architectures this is often confused with

- **Sliding-window / local attention** is not an alternative to multi-head; it
  is a *mask* (§1.5) applied inside each head. Combined freely with GQA.
- **Linear attention** replaces the softmax with a feature map so that the
  `(n, n)` matrix can be avoided via associativity — different mechanism,
  different file (§4.5).
- **State-space models** (Mamba, RWKV) are not attention at all: they are
  recurrences with a fixed-size state. Their selling point is exactly that they
  have no KV cache; their cost is exactly the recurrent bottleneck described in
  §0 of file 01.
- **Cross-attention** is the same operator with `q` from one sequence and
  `k, v` from another. Encoder-decoder models use it; decoder-only models
  today do not, having replaced it with concatenation of the two sequences in
  one causal stream.

---

Continue to [`03-in-the-transformer.md`](03-in-the-transformer.md) — where
attention sits inside a real model.