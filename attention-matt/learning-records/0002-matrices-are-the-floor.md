# Matrices are the floor, not a footnote

The learner asked for a matrices lesson before attention — unprompted, one session after the mission interview recorded "matrix maths recalled, not fluent" and accepted the plan of going straight to attention.

**Why this matters:** the original floor was set too high in a specific way. Recalling *that* a matrix multiply pairs rows with columns is not the same as being able to write it in a loop, or to predict the shape of a product before running it — and shape prediction is the one skill every later lesson silently assumes. Every remaining lesson in the mission (masking, multi-head, KV cache, reading nanoGPT) is a shape first and a mechanism second.

**Consequences for future sessions:**

- `lessons/0000-matrices-for-attention.html` is now the entry point; lesson 01 links to it as an explicit prerequisite. Numbering stays as the user asked for it ("lesson 0").
- `reference/matrix-ops.html` and `reference/glossary.html` exist so later lessons can cite the shape rule and the vocabulary instead of re-deriving them. The glossary is now the single source of truth for words like *score*, *weight*, *causal mask*, *head*.
- The feedback loop stays "your code vs an independent reference implementation" plus a named diagnosis of the specific wrong answer. The learner accepted that loop without explanation in lesson 01, and lesson 00 uses it for all three parts (dot, matmul, transpose).
- Do **not** assume shape fluency when the first new-axis shape appears. The causal mask is still (T × T); the batched `(B, H, T, d_head)` form that lessons 03+ need has not been taught, and 3Blue1Brown's column-stacking convention is a known confusion source (it is flagged in lesson 00 §1 and on the matrix-ops sheet).

**Evidence:** direct request, 2026-10-21: *"I need a lesson 0 that explains matrices."* It arrived after lesson 01 had been handed over but before any learner reaction to lesson 01 was recorded — so this is a self-assessment by the learner, not a diagnosis of a failed exercise. Lesson 01's own hand-off (did the axis diagnosis fire? did part A pass first try?) is still unknown and must be asked about rather than assumed.
