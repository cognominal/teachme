# Prior knowledge disclosed: strong engineer, no Python, matrices recalled

The learner is fluent in TypeScript, JS, Perl and C, has never written Python, and recalls matrix maths without being fluent in it. They are an everyday LLM user with little theory. They chose a browser sandbox as the code substrate with PyTorch shown alongside as reading, and 10–20 minute sessions.

**Why this matters:** it sets the floor for every lesson. Python syntax must be annotated, not assumed (`@`, `dim=`, `.shape[-1]`, indentation-as-block). One-line matrix idioms can be used; derivations cannot. Testing a function's output against a reference was accepted immediately and without explanation, so lessons can use "you impl vs reference impl" as the default feedback loop. The `.venv` in this workspace has CPU-only PyTorch 2.14.1 for sessions that run real code.

**Evidence:** stated directly in the mission interview (2026-10-21), then confirmed by the shape of the answers given — assistive detail on JS-vs-Python syntax was requested, not on functions or arrays.
