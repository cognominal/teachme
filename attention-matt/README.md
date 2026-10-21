# attention-matt

A session-based tutor for transformer attention, built with Matt Pocock's `/teachme`
skill. It is not a course you read: each session is one self-contained HTML page, the code
you write lives in `exercises/000N_*.py`, and a local grader checks that code, prints the
numbers it produces, and tells you the *specific* thing that is wrong when it does not.

- `MISSION.md` — who this is for, the constraints, and what "done" looks like.
- `lessons/000N-*.html` — one page per session: prose, figures, a retrieval quiz.
- `exercises/000N_*.py` — the code you edit, shipped as stubs with the step list in the
  docstring. Reference solutions sit in `exercises/solutions/` (never next to your file).
- `tools/grader/` — the grader: per-lesson checks, a CLI, and the HTTP server below.
- `reference/` — keepable one-pagers: the attention formula, matrix ops, Python for JS
  developers, a glossary.
- `HOW.md` — how the machinery works: page anatomy, the request → report → render flow, the
  checks contract, and the checklist for adding a lesson.
- `NOTES.md`, `RESOURCES.md`, `learning-records/` — working notes. `learning-records/` is
  untracked, so a clone may not have it at all (see [Learning records](#learning-records)).

---

## Quick start

```bash
python -m venv .venv            # once
./.venv/bin/pip install torch   # CPU build, ~200 MB. Nothing else is needed.
./bin/grader                    # serve the pages + the grading endpoint
./bin/lesson 0001               # open lesson 01
```

Then: read the page, edit the exercise file it names, click **Run the checks on my file**,
and read the verdicts followed by **Your file's own output** — the real stdout of your
script, traceback and all while it is still a stub.

---

## Running the server that serves the pages

```bash
./bin/grader                    # http://127.0.0.1:8757
./bin/grader --port 9000        # anywhere else
```

`./bin/grader` is a thin wrapper around `tools/grader/serve.py`. It does two things:

1. **Serves this whole directory as static files**, so every lesson page has a URL:

   | Lesson | URL under `http://127.0.0.1:8757/` |
   |---|---|
   | 0000 — matrices for attention | `lessons/0000-matrices-for-attention.html` |
   | 0001 — attention as a weighted average | `lessons/0001-attention-as-a-weighted-average.html` |
   | 0002 — the mask and the scale | `lessons/0002-the-mask-and-the-scale.html` |
   | 0003 — several weighted averages | `lessons/0003-several-weighted-averages.html` |

   `./bin/lesson 0003` finds the file, checks whether the server is up, and opens the URL
   through it (`xdg-open`). If the server is not running it opens the file from disk instead
   and says so.

2. **Answers the page's own requests** — this is why the server exists at all:

   | Route | Returns |
   |---|---|
   | `GET /health` | `{"ok": true, "lessons": ["0001", "0002", "0003"]}` |
   | `GET /grade?lesson=0003` | the same report `./bin/grade 0003 --json` prints |

   A lesson page's button does `fetch("/grade?lesson=000N")` from the same origin, so the
   verdicts, the figures and your file's output are rendered inline. That is also why
   `./bin/lesson` prefers the served URL over the local file.

Things worth knowing:

- **Binds `127.0.0.1` only.** It is not reachable from the network, and it never executes
  code that arrived over the network: the only file graded is the one the lesson registry
  names under `exercises/`. Your file runs as a subprocess with a 20-second timeout and
  truncated output.
- **Restart it after adding a lesson.** The lesson registry is read at import time, so a new
  entry will not appear in `/health` until you stop and start `./bin/grader`.
- **A page opened from disk still works.** With `file://` and no server, the exercise panel
  shows an offline box naming the commands to run instead of pretending to check anything;
  the prose, figures and quiz are all still usable.
- `GRADER_PORT=9000 ./bin/lesson 0003` makes `bin/lesson` look on the port you chose.

---

## Grading without the browser

```bash
./bin/grade 0003                                    # grade exercises/0003_multihead.py
./bin/grade 0003 --file exercises/solutions/0003_multihead.py   # grade any file
./bin/grade 0003 --json                             # the raw report the server returns
./bin/grade 0003 --show-solution                    # then read the reference solution
```

The CLI prints the same verdicts, the same diagnoses and the same "your file's own output"
block that the page shows — one report, two front ends, so the terminal and the page can
never disagree.

`tools/check_lessons.py` is the other useful command: it parses every inlined `<script>` in
every lesson and every asset with node or bun, so a stray quote fails in the terminal instead
of in the browser.

```bash
./.venv/bin/python tools/check_lessons.py       # ✓ 10 file(s) parse cleanly
```

---

## Learning records

`learning-records/` holds short records of what you have actually demonstrated — what is now
known, and why it changes what gets taught next. They are **not tracked** (they are notes
about a person, not course material), so a fresh clone does not have the directory at all.
Nothing breaks when it is missing; the only script that touches it creates it on demand:

```bash
./bin/record "Masks click once you see the triangle"        # -> learning-records/0004-masks-click-...md
./bin/record "..." --evidence "graded 17/17 first try" --implications "skip the mask re-drill"
./bin/record --list                                         # what is already recorded
```

Numbering is `0001`, `0002`, … by scanning what is there, so it keeps counting across sessions
and starts at `0001` in a directory that does not exist yet.

---

## Recreating the environment

`.venv/` is not tracked (see the repository `.gitignore`) and there is no lockfile:

```bash
python -m venv .venv
./.venv/bin/pip install torch
```

Measured here: Python 3.13.16, `torch 2.14.1+cpu`, CPU only — no CUDA, no GPU code anywhere
in the workspace. `numpy` is present in this venv but nothing in the workspace imports it;
torch is the whole dependency.

---

## Troubleshooting

| Symptom | What it is |
|---|---|
| Page shows "the grader is not running…" | Start `./bin/grader` and reload. The page works offline; it just cannot check your code. |
| `Address already in use` | Another `./bin/grader` is already up — that *is* the server, use it, or `--port 9000`. |
| `./bin/grade` fails with `No such file or directory` | The venv is missing: `python -m venv .venv && ./.venv/bin/pip install torch`. |
| `/health` does not list the lesson you just added | The registry is read at import: restart `./bin/grader`. |
| A new lesson's page 404s | Same cause as above; the server was started before the file existed. |
| `no lesson starting with 0004` | Lesson ids are the file prefix in `lessons/`; list them with `ls lessons/`. |
| Lesson 0000 has no exercise panel | On purpose: it is JavaScript-only, and predates the Python substrate. |

Everything in this file that could instead be checked by running something is checked — see
`HOW.md` §8 for where the trust boundary sits.
