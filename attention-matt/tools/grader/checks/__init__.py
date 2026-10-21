"""Lesson checks for this workspace.

Each module exposes:

    PARTS = [
        {
            "id": "weights",
            "title": "Part A — attention_weights(Q, K)",
            "entry": "attention_weights",      # the name the learner must define
            "missing_hint": "...",             # shown when that name is absent
            "tests":   [{"name": "...", "run": fn(learner_fn) -> (ok, detail)}],
            "diagnose": [{"when": fn(learner_fn) -> bool, "hint": "..."}],
        },
    ]

    def figures(namespace) -> dict:  # numbers for the lesson page to draw

and imports anything it needs from `.support`.
"""
