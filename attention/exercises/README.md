# Exercises

Eleven things to build, in order. Solutions are in `solutions.py` and are
self-checking — run `.venv/bin/python exercises/solutions.py`. Each solution
states the invariant it verifies, so you can tell whether yours is right.

Do them without reading `attn.py` first. The point is to derive the shapes and
the failure modes rather than recall them.

---

**1. Numerically stable softmax.**
Implement `softmax(z)` yourself. Verify: rows sum to 1; `softmax(z) ==
softmax(z + c)` for `c = ±137`; and it returns finite values for logits of
magnitude 1000 in float32. Then *break* it on purpose by removing the max
subtraction and find the magnitude at which float32 produces `nan`.

**2. Scaled dot-product attention.**
Implement `attn(Q, K, V, mask)` with only `@` and `np.exp`. Support a boolean
mask where `True` means visible, and a scale argument. Verify against
`attn.scaled_dot_product_attention` for `n ≠ m`, and verify that rows of the
weight matrix sum to 1 and are exactly 0 on masked entries.

**3. Where `√d_k` really comes from.**
Sample `q, k ~ N(0, I_d)`. Empirically confirm `Var(q·k) = d_k`. Then, for
`d_k ∈ {16, 64, 256, 1024}` and 64 random keys, compute the mean softmax entropy
with and without the `1/√d_k` scaling. Find the smallest `d_k` at which the
unscaled entropy drops below 1 nat.

**4. Permutation equivariance, for the whole layer.**
Take a full `MultiHeadAttention`, permute the input rows, and verify
`forward(permuted x)[:, perm] == forward(x)` to 1e-12, including `W_o`. Then
turn on RoPE and show the property fails.

**5. Padding masks.**
Build a mask combining causality with a padding mask that hides key 5 from
every query. Verify: (a) column 5 of the weight matrix is exactly zero;
(b) changing the *content* of token 5 changes the output nowhere.
Then construct the case where a row is entirely masked and confirm you get NaN —
and write the two lines that would prevent it.

**6. RoPE from scratch.**
Implement rotary embedding via complex numbers (`z * exp(i·θ·p)`). Verify it
matches the real-valued rotation formulation. Verify `<rope(q,m), rope(k,n)>`
depends only on `m-n`. Then show the *aliasing* failure: with `θ_0 = 1`, a
rotation by `2π` is the identity, so positions `p` and `p + 2π` are
indistinguishable to the first frequency pair. What is `2π` in tokens?

**7. Online softmax.**
Implement blockwise attention with running max/denominator/numerator. Verify it
equals the naive result for `block=1`, `block=7` on `n=100`, and `block=64` on
`n=64`. Then instrument it: how many floats does the largest intermediate tensor
hold, for `n = 4096` and `block = 128`, versus the naive version?

**8. KV-cache budget.**
A device has 40 GiB free for cache. You serve a 32-layer, `d_model=4096` model
with 8 KV heads of dim 128, bf16, at 128k context. How many concurrent
sequences fit? How many if it had 32 KV heads instead of 8?

**9. ALiBi is purely relative.**
Implement ALiBi (`-slope_h · |i-j|` added to the logits). Verify the resulting
weight matrix is **Toeplitz** — constant along every diagonal — which is exactly
the statement "the weights depend only on the distance". Compare with an
attention matrix where each key carries a learned absolute position vector.
Explain in one sentence what a Toeplitz attention matrix cannot express.

**10. How many heads does a target need?**
With `H` heads sharing the same `V` and an identity-like `W_o`, the layer's
output is `Σ_h Σ_j a^(h)_j V_j = Σ_j w_j V_j` where `w_j = Σ_h a^(h)_j ≥ 0` and
`Σ_j w_j = H`. So the reachable set is exactly `H · conv(V rows)`.
Given `V = [[10,0],[0,10],[0,0]]`, compute the minimum `H` needed to reach
`[10,10]`, then to reach `[10,10,10]` from
`V = [[10,0,0],[0,10,0],[0,0,10]]`. Confirm both by search.

---

## Conceptual questions

**A.** A single attention layer with *content-independent* weights reduces to
`A V` for a fixed matrix `A`. What is the maximum expressivity of such a layer,
and what exactly is the softmax contributing?

**B.** Why does `1/√d_k` use `d_k` (the key width) and not `d_v` or `d_model`?

**C.** You have a decoder with one head and a 4096-wide `d_model`. You want a
128-wide `d_h`. What happens to `W_q`'s shape, and what does that mean for the
`num_heads` you report?

**D.** A model's attention maps show 40% of heads putting more than half their
mass on token 0. Is the model broken? What is the mechanism, and what happens if
you evict token 0's KV entry from a sliding window?

**E.** Why is teacher-forced training with a causal mask valid, while the same
trick with a bidirectional mask would be training on the answers?

**F.** You must cut inference cost in half. You have two levers: prune half the
attention heads, or halve the number of KV heads (GQA), or use a sliding window
of half the context. Which reduces *memory traffic* most per unit of quality
lost, and why?

**G.** "Attention is `O(n²)`, so a 1M-token context is impossible." What are the
three distinct ways one could get around that, and what does each one give up
mechanically?