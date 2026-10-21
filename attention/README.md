# Attention, end to end

A self-contained course on how attention works in an AI model: the mathematics,
the implementation, where it sits inside a transformer, why it is expensive, and
what today's models actually do. Everything here runs, and most of the claims are
checked numerically in front of you rather than asserted.

```
teachme/attention/
├── notes/                  the reading
│   ├── 01-mechanics.md         what attention computes, and why
│   ├── 02-multi-head.md        H routers, GQA, what heads learn
│   ├── 03-in-the-transformer.md  blocks, training, KV cache, cost
│   └── 04-modern.md            flash attention, variants, long context, configs
├── code/
│   ├── attn.py                 the whole mechanism, from scratch in NumPy
│   ├── lab.py                  13 runnable demonstrations that print numbers
│   ├── verify.py               30 checks against PyTorch + invariants
│   └── tiny_transformer.py     train a real transformer, inspect its heads
├── exercises/                  problems + verified solutions
└── .venv/                      numpy + torch (already installed)
```

## Read it in this order

| if you have | do this |
|---|---|
| 15 minutes | `lab.py softmax scale shapes causal`, then §1.1–1.5 of note 01 |
| 1 hour | all of note 01 + `lab.py` in full, skimming the output |
| 3 hours | notes 01 → 04 in order, running each lab as it comes up |
| a weekend | add `tiny_transformer.py` (train both tasks) and `verify.py`; do the exercises |

Do not read the notes without running the labs. Every lab prints the numbers it
is talking about, and several of them will contradict your intuition — that is
what they are for.

## Run things

```bash
.venv/bin/python code/lab.py                  # all 13 labs
.venv/bin/python code/lab.py scale causal      # just those
.venv/bin/python code/verify.py                # 30 numerical checks
.venv/bin/python code/tiny_transformer.py      # ~90s: trains two models
.venv/bin/python exercises/solutions.py        # exercise answers, self-checking
```

The venv has `numpy 2.5.4` and `torch 2.14.1+cpu` (Python 3.12). Torch is used
for exactly one purpose — as an *independent* implementation to check the NumPy
one against, and to train the toy transformer. Nothing in `attn.py` imports it.

## The mechanism, in one screen

```
Attn(Q, K, V) = softmax( Q Kᵀ / √d_k + M ) V

  Q  (n_q, d_k)   what each position is looking for      M  mask: -inf where forbidden
  K  (n_kv, d_k)  what each position advertises              (causal, padding, window)
  V  (n_kv, d_v)  what each position hands over
  A  (n_q, n_kv)  the weights: rows non-negative, sum to 1
```

That is the entire mechanism. It has not changed since 2017. Everything after it
in the literature is about the cost of `A`, the memory cost of `K` and `V`, or
about where positional information enters.

## The seven things most explanations get wrong

1. **"Attention is a lookup of stored knowledge."** No — the output is a
   *convex combination of value vectors*. Attention only routes and mixes.
   Knowledge lives in the parameters; attention decides who gets to read it.
2. **"√d_k is a tuning constant."** It is a variance correction: unscaled, the
   logit spread grows as `√d_k`, the softmax goes one-hot, and its Jacobian
   (`p(1-p)`) collapses to zero, so the head stops learning. Lab 2 measures it.
3. **"Attention knows about order."** It does not. Permute the input and the
   output multiset is identical — verified to 4e-16 in lab 6. Order comes from
   RoPE / PE / ALiBi or not at all.
4. **"More heads = more heads of the same thing."** Each head has its *own*
   `W_q, W_k, W_v`, so it is a genuinely independent router. `W_o` then mixes
   them linearly, so heads can also cancel each other.
5. **"Attention is the expensive part."** Below ~4×`d_model` tokens, the MLP has
   it beat on FLOPs by a wide margin. Attention's quadratic term only dominates
   for long contexts — which is why long-context is attention research.
6. **"Flash attention is an approximation."** It is exact. Online softmax is an
   algebraic identity, checked to 6.7e-16 in lab 9.
7. **"It's just scaled dot-product attention everywhere."** The KV side is
   usually GQA or MLA, the positional scheme is RoPE with a rescaled base, and
   long-context models alternate local and global layers. The mechanism is the
   same; the plumbing is not.

## What is actually verified here

`code/verify.py`, 30 checks, all passing:

- the NumPy implementation against `torch.nn.functional.scaled_dot_product_attention`
  across head widths, non-square `n ≠ m`, causal masks, additive ALiBi bias, and
  a hand-rebuilt multi-head layer;
- the three-loop textbook definition against the vectorised one, bit-for-bit;
- causality (a future perturbation cannot change a past output — `np.array_equal`,
  not `allclose`);
- permutation equivariance, and its failure once RoPE is enabled;
- RoPE's relative property at offsets from 3 to 2²⁰, its equivalence to the
  complex-exponential formulation, and norm preservation;
- flash attention against naive attention, including a padding mask and
  non-divisible block sizes;
- KV-cache incremental decoding against the parallel forward pass;
- the published KV-cache figure for Llama-2-7B at 4096 context = exactly 2 GiB;
- softmax shift invariance, the float32 overflow that motivates it, and the
  `p(1-p)` collapse;
- that a fully masked row produces NaN, on purpose, so you know the failure mode.

`code/tiny_transformer.py` trains two real models from scratch and prints their
attention maps — a previous-token head reaching 100% held-out accuracy, and an
induction head measured by *pointer accuracy* (does its argmax land on the value
that follows the earlier occurrence of the query token?).

## Sources worth reading next

- Vaswani et al. 2017, *Attention Is All You Need* — the original; §3.2 is the
  mechanism and the rest is the architecture.
- Bahdanau et al. 2015 — additive attention, for contrast.
- Su et al. 2021, *RoFormer* — RoPE.
- Dao et al. 2022, *FlashAttention*.
- Shazeer 2019 (MQA), Ainslie et al. 2023 (GQA), DeepSeek-V2 (MLA).
- Olsson et al. 2022, *In-context Learning and Induction Heads* — heads doing
  something legible.
- Sanford, Hsu, Telgarsky 2023 — what attention can and cannot represent.
- Xiao et al. 2023, *Efficient Streaming Language Models with Attention Sinks*.
- Elhage et al. 2021, *A Mathematical Framework for Transformer Circuits* — the
  residual-stream-as-bus picture used in note 02.