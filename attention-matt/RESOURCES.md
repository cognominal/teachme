# Transformer Attention Resources

Curated for: implementing self-attention from scratch in code, reading real model code, with minimal maths and 10–20 minute sessions. Every URL here was fetched and checked; sources that failed to load were dropped rather than guessed at.

## Knowledge

### Python &amp; PyTorch substrate (from lesson 01 on)

All three reachable and checked 2026-10-21. These support the code loop rather than the theory.

- [The Python Tutorial](https://docs.python.org/3/tutorial/) ([§4 control flow and functions](https://docs.python.org/3/tutorial/controlflow.html))
  The canonical language reference for a reader who is fluent in JS/Perl/C but new to Python: blocks as indentation, `def`, `for … in range(n)`, tuple unpacking, comprehensions. **Use for:** any syntax question a lesson's inline annotations did not settle. The workspace's own compressed map is `reference/python-for-js-devs.html`.
- [PyTorch — Tensors tutorial](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
  Shapes, dtypes, `@` vs `*`, broadcasting, and moving between tensors and Python lists, in the library's own words. **Use for:** "what does this tensor operation actually do to the shape".
- [PyTorch docs — `torch.softmax`](https://docs.pytorch.org/docs/stable/generated/torch.softmax.html)
  The `dim` argument and the numerical-stability note. **Use for:** the axis question in attention, which is the most common implementation bug in the field.

### The maths floor (matrices, shapes, axes)

- [3Blue1Brown, _Essence of Linear Algebra_ — ch. 4, "Matrix multiplication as composition"](https://www.3blue1brown.com/lessons/matrix-multiplication/) (watch after [ch. 3, "Linear transformations and matrices"](https://www.3blue1brown.com/lessons/linear-transformations/))
  The _why_ behind rows-against-columns: composing two transformations, and applying a matrix to every column of the right operand. Each page carries the full text adaptation, so it reads as well as it watches. **Use for:** the intuition behind the shape rule, before or after lesson 00. **Trap it sets (flagged in lesson 00, §1):** it stacks vectors as _columns_ and works in 2-D; this workspace's grids are its transpose.
- [Dive into Deep Learning §2.3, _Linear Algebra_](https://d2l.ai/chapter_preliminaries/linear-algebra.html)
  Textbook ramp: scalars → vectors → matrices, dot products, matrix–vector and matrix–matrix products, transposes, and batched matmul, with PyTorch alongside each. **Use for:** the mechanical rule in written form, and for batched shape rules when the head/batch axes arrive.
- [PyTorch docs — `torch.matmul`](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html)
  The exact broadcasting rules for leading axes: which dimensions must match, which broadcast, what the result shape is. **Use for:** any shape question about `(B, H, T, d)` tensors. Claims used in lesson 00 were run against this workspace's torch 2.14.1 on 2026-10-21: `(2,3) @ (3,2) → (2,2)`; `(2,4,8) @ (2,8,4) → (2,4,4)`; 1-D `@` 1-D → a 0-dim tensor; a mismatch reads `mat1 and mat2 shapes cannot be multiplied (2x3 and 2x3)`; `.T` on a 3-D tensor reverses _all_ axes and now emits a deprecation warning — `transpose(-2, -1)` is the correct spelling.

### First contact / intuition

- [3Blue1Brown, _Attention in transformers, step-by-step_ (Deep Learning Ch. 6)](https://www.3blue1brown.com/lessons/attention/)
  Grant Sanderson. 26-minute video with a full text transcript on the page. Best visual treatment of self-attention: Q/K/V as question/answer, the score grid, masking, multi-head. **Use for:** intuition before or after any lesson. **Trap it flags itself:** it draws tokens in *columns* and softmaxes down them; the paper (and this workspace) use rows. Identical maths, transposed picture.
- [Jay Alammar, _The Illustrated Transformer_](https://jalammar.github.io/illustrated-transformer/)
  Diagram-first walkthrough of the whole architecture — multi-head, positional encoding, residuals, layer norm. **Use for:** seeing where attention sits in the block, once the mechanism itself is clear.
- [Jay Alammar, _Visualizing a Neural Machine Translation Model_](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
  The pre-history: the fixed context-vector bottleneck attention was invented to fix. **Use for:** answering "why does attention exist at all".
- [Hugging Face LLM Course, Ch. 1.4 — How do Transformers work?](https://huggingface.co/learn/llm-course/chapter1/4)
  Plain-language orientation to encoder / decoder / encoder-decoder families and masking. **Use for:** vocabulary before reading anything else.

### Canonical formulas (cite these, never memory)

- [Vaswani et al., _Attention Is All You Need_ (2017)](https://arxiv.org/abs/1706.03762)
  §3.2.1 defines scaled dot-product attention; footnote 4 is the scaling argument; Appendix B.1 the positional encoding. **Use for:** every formula claim in a lesson. Also on arXiv as HTML: `https://arxiv.org/html/1706.03762v7`.
- [Dive into Deep Learning, §11.3 Attention Scoring Functions](https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html)
  Derives the variance argument for `1/√d` properly, contrasts dot-product and additive scoring, and shows masked softmax for padding. **Use for:** the exact reason the scale exists, when the lesson's plain-English version is not enough.
- [Jurafsky & Martin, _Speech and Language Processing_ (3rd ed. draft), Ch. 9](https://web.stanford.edu/~jurafsky/slp3/9.pdf)
  Textbook-grade statement of `softmax(QKᵀ/√d_k)V` and the causal mask as a `−∞` upper triangle. **Use for:** a second authoritative phrasing of the same equations.
- [Lilian Weng, _Attention? Attention!_](https://lilianweng.github.io/posts/2018-06-24-attention/)
  One-page index of the whole family: additive (Bahdanau), dot-product, scaled dot-product, self-attention, soft vs hard, global vs local. **Use for:** seeing the lineage in a single table.
- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/)
  The paper's text interleaved with line-by-line PyTorch. **Use for:** the bridge from formula to runnable code, in a session where we read real Python.

### Code to read and run

- [PyTorch — `torch.nn.functional.scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
  The production kernel, with the equivalent hand-written code in the docs, mask semantics, `is_causal`, and GQA. **Use for:** what real implementations call, and what they add.
- [PyTorch — `torch.nn.functional.softmax`](https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html)
  `dim` semantics and the numerical-stability note. **Use for:** the normalisation step in isolation, especially the axis question.
- [PyTorch — `torch.einsum`](https://pytorch.org/docs/stable/generated/torch.einsum.html)
  Writing attention as an indexed equation. **Use for:** when shape bookkeeping across heads gets confusing.
- [karpathy/nanoGPT — `model.py`](https://github.com/karpathy/nanoGPT)
  A complete GPT in ~300 lines, attention included. **Use for:** lesson 03+, reading real model code. Karpathy now points to [nanochat](https://github.com/karpathy/nanochat) instead; nanoGPT is deprecated but remains the clearest teaching artefact.
- [Karpathy, _Let's build GPT: from scratch_](https://www.youtube.com/watch?v=kCc8FmEb1nY) (and the [Zero to Hero series](https://karpathy.ai/zero-to-hero.html))
  Builds a working GPT line by line, live. **Use for:** the long watch, when a 20-minute session becomes 2 hours.
- [PyTorch — `torch.nn.MultiheadAttention`](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
  The `MultiHead(Q,K,V) = Concat(head_i)W^O` definition plus mask/padding semantics (note: its boolean-mask polarity is inverted relative to SDPA). **Use for:** multi-head in lesson 04.

### This workspace's own material

- `lessons/0000-matrices-for-attention.html` — dot product, matmul, transpose, the shape rule, a shape-prediction drill, and the `Q · Kᵀ` grid built from the learner's own functions. (JavaScript arrays: this is the one lesson before Python.)
- `lessons/0001-attention-as-a-weighted-average.html` — scaled dot-product attention in PyTorch: the four steps, the Python needed, the softmax axis, why the `1/√d_k` scale exists. Its verdicts come from the local grader.
- `exercises/` — the learner's implementations, one file per lesson, plus `exercises/solutions/` with reference solutions.
- `bin/` + `tools/grader/` — the loop: `bin/grade <lesson>` grades `exercises/<lesson>_*.py` against independent reference implementations with named diagnoses; `bin/grader` serves the workspace and the same report over http for the lesson pages; `bin/lesson <id>` opens a lesson through it. Checks per lesson live in `tools/grader/checks/lesson_NNNN.py`.
- `tools/check_lessons.py` — parses every inlined lesson script and every shared asset with node, so a stray quote is caught in the terminal instead of in a browser.
- `reference/scaled-dot-product-attention.html` — one-page sheet: four steps, shapes, the PyTorch/JS mapping, four failure modes, hand-checkable sanity tests.
- `reference/matrix-ops.html` — one-page sheet: the three operations, the inner-numbers rule, `Q · Kᵀ` and `weights · V`, the `.T` trap, four shape errors.
- `reference/python-for-js-devs.html` — the syntax map, the tensor idioms, the error messages you actually hit, and the workspace's commands.
- `reference/glossary.html` — every term this course uses, defined once; the single source of truth for vocabulary.

## Wisdom (Communities)

All verified reachable and active (2026-10-21). Every option is free. The honest ordering for *this* learner — a strong engineer early in the theory — is: start at the bottom, not the top. Ask in the beginner-appropriate places first.

**Start here**

- [r/MLQuestions](https://www.reddit.com/r/MLQuestions/) and [r/learnmachinelearning](https://www.reddit.com/r/learnmachinelearning/)
  Purpose-built for beginner questions; questions-only, no ads. **Use for:** "here is my derivation of the `1/√d_k` scale — where is it wrong?" Answer quality varies, so bring a specific artefact.
- [Hugging Face Forums](https://discuss.huggingface.co/)
  Discourse forum, threads answered within hours; maintainers cite the code. **Use for:** concrete implementation bugs — "where exactly is the causal mask applied, and why do logits differ between eager and SDPA?"
- [PyTorch Forums](https://discuss.pytorch.org/)
  Official user forum, active daily. **Use for:** tensor/autograd mechanics — "is my scaled dot-product implementation applying the mask *before* softmax?" The single best place to test a from-scratch implementation against people who know the framework.

**Once there is something to show**

- [ML Collective — Open Collab](https://discord.gg/U2MzJYGS9P) ([site](https://mlcollective.org/community/))
  Non-profit mentorship community: weekly DLCT reading group since 2018, Research Jam, **1:1 Office Hours** with experienced researchers. **Use for:** critique of a written-up replication plan, and the office hours. This is the highest-value single community for someone teaching themselves with a real project.
- [GPU MODE](https://discord.gg/gpumode) ([site](https://gpumode.com))
  GPU-programming community with a kernel leaderboard, 100+ recorded lectures, recurring competitions. **Use for:** when attention becomes a performance problem — benchmark your kernel against the reference and against humans.
- [AI Tinkerers](https://aitinkerers.org/all_cities) — 268 city chapters, demo-first ("No slides. Demo the thing.")
  **Use for:** in-person demo of a from-scratch attention implementation. Free, RSVP-based.

**Deep / specialist (bring a repro, not a question)**

- [EleutherAI Discord](https://discord.com/invite/eleutherai)
  The open-science lab's coordination space. **Warning, from their own FAQ:** *"We are a research-focused discord server and not an educational one. We welcome beginners to lurk… but this is not the place to get intro-level resources or answers to basic questions."* **Use for:** lurking, and narrow questions tied to something you built. Also runs [SOAR](https://www.eleuther.ai/soar), a free mentored research program.
- [Mech Interp Discord](https://discord.gg/TnTGx7cQwh)
  Single-topic: mechanistic interpretability and replications. **Use for:** "here is my head map from a reimplemented GPT-2 attention — is my QK/OV composition right?"
- [Cohere Labs Open Science Community](https://cohere.com/research/open-science)
  Subfield groups inc. ML Math & Theory and **BIRDS (Beginners in Research Driven Studies)**. Gated (free): apply; review weekly, so budget about a week.
- [Alignment Forum](https://www.alignmentforum.org/) (posting is members-only) → publish on [LessWrong](https://www.lesswrong.com/) first and let it be promoted up.
  **Use for:** the interpretability frontier, and getting a write-up critiqued.
- [Papers We Love](https://paperswelove.org/chapter) — ~50 chapters of CS paper reading groups. **Use for:** reading-group discipline on the original papers.
- [vLLM Forums](https://discuss.vllm.ai/) — run by the people who wrote PagedAttention. **Use for:** KV-cache and serving internals, once the mission reaches inference.

**Not verified / caution**

- LMSYS Discord and r/LocalLLaMA Discord invites: both published invites resolve to Discord's generic landing page — treat them as dead.
- Reddit blocks this workspace's reader (HTTP 403), so subreddit rules above come from Reddit's own public rules payloads rather than a direct fetch.
- Static Slack invites (PyTorch, SGLang) rotate; always re-copy from the official site footer.

## Gaps

Nothing verified yet for these; future research passes should fill them:

- **KV cache mechanics** and why inference reuses keys/values — the mission cares about this, no single high-trust explainer confirmed.
- **Interpretability of learned attention patterns** (attention heads as circuits, induction heads) — `transformer-circuits.pub` not yet verified in this workspace.
- **Multi-head in practice:** how heads are split from the width, and what GQA/MQA actually save.
- **Batched shape notation:** how `(B, H, T, d_head)` arises. The *mechanics* are now covered (d2l §2.3 and the `torch.matmul` docs, both verified 2026-10-21); what is still missing is a high-trust source that derives the head axis from a real model's weight split.
- Local/in-person options: no verified reading group or meetup confirmed for this user's location.
