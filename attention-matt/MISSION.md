# Mission: Transformer attention, implemented from scratch

## Why
You use LLMs daily but the machinery underneath is a black box you cannot inspect. You are a working engineer (TypeScript, JS, Perl, C) who wants attention to stop being magic: write it yourself, watch it produce correct numbers, and change it deliberately — masking, multi-head — instead of trusting a diagram.

## Success looks like
- Write scaled dot-product attention from memory — scores, softmax, weighted sum of values — with correct shapes for a batch of sequences.
- Say why each step exists (the `1/√d_k` scale, the softmax axis, the causal mask) and what breaks when you remove it.
- Read the attention block of a real model (nanoGPT-scale) line by line without help.
- Extend the core yourself: multi-head, causal mask, and explain what a KV cache reuses across steps.

## Constraints
- 10–20 minute sessions. Short lessons, one tangible win each.
- Fluent in JS/TS/Perl/C, new to Python — **from lesson 01 on, all code is Python + PyTorch**, executed for real against this workspace's CPU torch (`.venv/`, torch 2.14.1). Python syntax is annotated at first use and collected in `reference/python-for-js-devs.html`; JS appears only where it made shapes visible first (lesson 00). The loop is: edit `exercises/<lesson>.py`, run `./bin/grade <lesson>`; the lesson page fetches the same verdicts when `./bin/grader` is running.
- Matrix maths recalled, not fluent: notation stays minimal and is explained inline at first use.
- Cognitive difficulty goes into retrieval practice, not into arithmetic.

## Out of scope
- Training or fine-tuning models; tokenizers, datasets, evaluation harnesses.
- GPU kernels, FlashAttention internals, CUDA, distributed training.
- Cognitive or psychological attention.
