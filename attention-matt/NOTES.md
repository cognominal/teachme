# Notes

## Learner preferences (recorded 2026-10-21)

- Code-first. Wants runnable code, checked against a reference implementation.
- **Python + PyTorch is the substrate from lesson 01 on** (asked for directly: "rewrite the session 1 to use pytorch … introduced in the session 0"). Not a Python speaker: annotate syntax at first use, keep the crib sheet at `reference/python-for-js-devs.html` current, and never assume `@`, `dim=`, `.shape`, indentation-as-block or `torch.` prefixes are obvious.
- Math: minimal, notation explained inline at first use. Wants the *reason* for the `1/√d_k` scale, not the derivation. Phrasings that have landed: "inner numbers must match", "one row = one token", "matmul applies B to every row of A", "a lookup that returns a blend".
- Sessions: 10–20 minutes. One tangible win per lesson, mission anchor "implement attention and have it work" — every lesson ends with something running and producing correct numbers.
- **Asks for foundations himself.** Session 2: "lesson 0 that explains matrices". Session 3: switch to PyTorch. Treat a self-declared gap as a fact, not a guess (see `learning-records/0002`, `0003`); number such a lesson as the learner names it (`0000`).

## Workspace conventions

- Teaching matrices are arrays of rows — `[[..],[..]]` is `(rows × cols)`, matching the `(T, d)` prose in the paper. In Python this is `Q.shape == (T, dk)`.
- **The implementation lives in a real file, not in the page.** `exercises/<lesson>_*.py` holds the learner's code (stubs with the step list in comments); `exercises/solutions/<lesson>_*.py` holds a reference solution. Lesson pages carry prose, diagrams and retrieval practice only.
- The loop: `./bin/grade 0001` (terminal report), `./bin/grade 0001 --show-solution`, `./bin/grader` (local service + static server on 127.0.0.1:8757), `./bin/lesson 0001` (opens the lesson through the service if it is up), `./.venv/bin/python exercises/0001_attention.py` (the learner's own numbers, in a table).
- **Checks live in `tools/grader/checks/lesson_NNNN.py`** and are shared by the CLI and the service, so the terminal and the page can never disagree. Each part declares `entry`, `tests` and `diagnose` (ordered specific → general, last one `lambda fn: True`). Reference implementations live in the same file, written independently of `exercises/`. Add a lesson to `core.LESSONS` when its checks exist.
- Diagnoses are authored once in the page's HTML; the CLI strips tags (`cli.plain`).
- Tolerance: float32 chains land within ~4e-8 of float64, so `support.ATOL = 1e-5`.
- Assets are shared; a lesson must not re-implement them: `assets/matrix-view.js` (every matrix on screen), `assets/shape.js` (`Shape.product`, `Shape.explorer`), `assets/pyexercise.js` (fetches `/grade` and draws the verdicts), `assets/exercise.js` + `assets/tensor.js` (lesson 00's in-browser JS harness), `assets/quiz.js`, `assets/lesson.css` (the only stylesheet).
- Before hand-off: `./.venv/bin/python tools/check_lessons.py` (parses every inline `<script>` and every asset with node — a stray quote in a lesson is otherwise only visible in a browser), then a Chromium smoke test of the real paths.
- **Browser smoke-testing gotcha:** `tab.run(async ({page}) => page.evaluate(...))` evaluates in an *isolated world*. Page globals are invisible (`PyExercise`, `Quiz`, `Exercise` are `undefined` there) and expando properties set by page code do not read back. Drive the real DOM instead: `#exercise .pyex-part`, `.feedback`, `.tests li`, `button.click()`, `textarea.value`, `data-passed`. DOM events do cross the boundary.
- Smoke tests worth repeating on any lesson change: open with the stub in place (expect failures + a named diagnosis), copy the solution over the learner file (expect "all parts pass" and matrices that say *your code*), restore the stub, then walk the quiz answer key by clicking.
- Grading the shipped stub must report `NotImplementedError`, not a crash; a solution in `exercises/solutions/` must pass via `--file` before it is offered to the learner.

## Open threads

- **Delivery order:** 00 matrices → 01 attention (Python) → 02 the scale + the causal mask → 03 multi-head ("several weighted averages over subspaces") → read `nanoGPT`'s attention block line by line → KV cache.
- **Lesson 02 is next, and is not authored yet.** Planned shape: one thesis — *the score matrix is the only control surface*; the scale and the mask are both edits to it before softmax. Python checks go in `tools/grader/checks/lesson_0002.py` (mask as `−∞` before softmax, why `0` would not do, saturation vs `d_k`); the page draws a lower-triangular weight grid via `MatrixView`'s `cellClass`.
- **Lesson 01's hand-off is still unverified** — no learner reaction has been recorded for either the old JS version or the new PyTorch one. Ask before assuming: did the checks pass first try, and did the diagnosis point at the right thing? That answer decides whether lesson 02 is masking or a row-vs-column re-drill.
- The learner has not yet been asked about communities. Sequencing stays r/MLQuestions + PyTorch Forums now, ML Collective Office Hours once there is a project; EleutherAI Discord is explicitly not for intro questions.
- Unused hooks ready for later lessons: `MatrixView` `cellText`/`cellClass` (∞/−∞ cells, masked cells), `Shape.explorer` presets (batched `(B, T, dk)`), `pyexercise.js` `intro` + `onResult` for a page that draws more than verdicts.
