# Substrate switched to Python + PyTorch (from lesson 01 on)

The learner asked for session 1 to be rewritten in PyTorch rather than JavaScript, immediately after lesson 00 introduced PyTorch as reading. All lesson-01 code, the checks and the feedback loop are now Python: the implementation lives in `exercises/0001_attention.py`, is graded by `./bin/grade 0001` against an independent reference implementation, and runs on this workspace's real CPU torch (2.14.1).

**Why it matters:** the mission's original constraint ("browser sandbox first, PyTorch alongside as reading") is superseded. Two of the mission's success criteria — *read the attention block of a real model line by line* and *extend the core yourself* — are Python activities, so the reading-to-writing gap had to close anyway; the learner closed it himself. It also means every future lesson needs a check module in `tools/grader/checks/`, not an in-page sandbox, and that Python syntax support is now part of teaching rather than incidental.

**Evidence:** direct request, 2026-10-21: *"still using the skill rewrite the session 1 to use pytorch (instead of js) introduced in the session 0."* No claim is made here about how well the new loop worked for the learner — the lesson has not been attempted yet.

**Implications:** still assume no Python fluency, but assume fluency in *reading* torch — the crib sheet (`reference/python-for-js-devs.html`) is now a first-class reference, and lessons may use `@`, `dim=`, `.shape` and `.transpose(-2, -1)` freely as long as each is annotated where it first appears. Lesson 00 stays in JavaScript deliberately: it carried the shape rule before any Python existed.

Mission updated: `MISSION.md` constraints now describe the Python loop.
