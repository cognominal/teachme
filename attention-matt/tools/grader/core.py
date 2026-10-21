"""core.py — run the learner's own Python against a lesson's checks.

One grader, two front ends: `bin/grade` (terminal) and `bin/grader` (local
service + static file server). They share this module, so the terminal verdicts
and the in-page verdicts can never disagree.

Nothing here executes code that arrived over the network. The only thing graded
is a file in `exercises/`, named by the lesson's spec.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
_GRADER_DIR = Path(__file__).resolve().parent
if str(_GRADER_DIR) not in sys.path:  # so `import checks.lesson_0001` works from anywhere
    sys.path.insert(0, str(_GRADER_DIR))

# One entry per lesson that has Python exercises.
LESSONS: dict[str, dict[str, Any]] = {
    "0001": {
        "title": "Attention is a weighted average",
        "source": "exercises/0001_attention.py",
        "checks": "checks.lesson_0001",
        "solution": "exercises/solutions/0001_attention.py",
    },
    "0002": {
        "title": "The mask and the scale",
        "source": "exercises/0002_masking.py",
        "checks": "checks.lesson_0002",
        "solution": "exercises/solutions/0002_masking.py",
    },
    "0003": {
        "title": "Several weighted averages at once",
        "source": "exercises/0003_multihead.py",
        "checks": "checks.lesson_0003",
        "solution": "exercises/solutions/0003_multihead.py",
    },
}


@dataclass
class PartResult:
    id: str
    title: str
    passed: bool = False
    verdict: str = ""
    tests: list[dict[str, Any]] = field(default_factory=list)
    diagnosis: str | None = None
    raised: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "passed": self.passed,
            "verdict": self.verdict,
            "tests": self.tests,
            "diagnosis": self.diagnosis,
            "raised": self.raised,
        }


def _clean_traceback(exc: BaseException, filename: str) -> str:
    """Keep only the frames that belong to the learner's file."""
    lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    kept: list[str] = []
    for chunk in lines:
        if filename in chunk or not kept:
            kept.append(chunk)
    return "".join(kept).strip()


def run_learner_source(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Execute the learner's file in a fresh namespace.

    Returns (namespace, error). `__name__` is not "__main__", so a demo block at
    the bottom of the file does not run during grading.
    """
    if not path.exists():
        return None, {"kind": "missing", "message": f"{path} does not exist yet", "line": None, "text": None}

    source = path.read_text()
    namespace: dict[str, Any] = {
        "__name__": "learner",
        "__file__": str(path),
        "__builtins__": __builtins__,
    }
    try:
        code = compile(source, str(path), "exec")
    except SyntaxError as exc:
        text = None
        if exc.lineno and exc.lineno <= len(source.splitlines()):
            text = source.splitlines()[exc.lineno - 1]
        return None, {
            "kind": "syntax",
            "message": f"{exc.msg} (line {exc.lineno})",
            "line": exc.lineno,
            "text": text,
        }

    try:
        exec(code, namespace)
    except BaseException as exc:  # noqa: BLE001 - the learner's code can raise anything
        return None, {
            "kind": "runtime",
            "message": _clean_traceback(exc, str(path)),
            "line": None,
            "text": None,
        }
    return namespace, None


def _safe(fn: Callable[[], Any]) -> tuple[bool, Any]:
    try:
        return True, fn()
    except BaseException as exc:  # noqa: BLE001
        return False, exc


DEMO_TIMEOUT = 20.0
DEMO_STDOUT_LIMIT = 6000
DEMO_STDERR_LIMIT = 2000


def run_learner_demo(path: Path) -> dict[str, Any]:
    """Run the learner's file the way *they* run it — as a script — and capture what it prints.

    The checks import the file (so a demo block guarded by `__main__` never fires);
    this runs it, which is what makes the page able to show the same table the
    terminal shows. The only path involved is the one the lesson named, so the
    trust boundary is unchanged: nothing that arrived over the network is executed.
    """
    empty = {"stdout": "", "stderr": "", "returncode": None, "timed_out": False, "truncated": False}
    if not path.exists():
        return empty

    timed_out = False
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=DEMO_TIMEOUT,
        )
        stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = _as_text(exc.stdout)
        stderr = _as_text(exc.stderr)
        code = None
    except OSError as exc:  # no interpreter, no permission, ...
        return {**empty, "stderr": f"{type(exc).__name__}: {exc}", "returncode": None}

    truncated = len(stdout) > DEMO_STDOUT_LIMIT
    if truncated:
        stdout = stdout[:DEMO_STDOUT_LIMIT] + "\n… (output truncated)\n"
    return {
        "stdout": stdout,
        "stderr": _fallback(stderr, DEMO_STDERR_LIMIT),
        "returncode": code,
        "timed_out": timed_out,
        "truncated": truncated,
    }


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


def _fallback(text, limit: int) -> str:
    return text if len(text) <= limit else text[-limit:]


def grade(lesson_id: str, figures: bool = True, source: Path | None = None, demo: bool = True) -> dict[str, Any]:
    """Grade one lesson and return a JSON-able report."""
    spec = LESSONS.get(lesson_id)
    if spec is None:
        return {"lesson": lesson_id, "error": {"kind": "unknown", "message": f"no lesson {lesson_id!r}", "line": None, "text": None}}

    source_path = Path(source) if source is not None else ROOT / spec["source"]
    report: dict[str, Any] = {
        "lesson": lesson_id,
        "title": spec["title"],
        "source": spec["source"] if source is None else str(source_path),
        "error": None,
        "parts": [],
        "figures": None,
        "output": run_learner_demo(source_path) if demo else None,
    }

    namespace, error = run_learner_source(source_path)
    if error is not None:
        report["error"] = error
        return report

    checks = importlib.import_module(spec["checks"])

    for part_spec in checks.PARTS:
        result = PartResult(id=part_spec["id"], title=part_spec["title"])
        entry = part_spec["entry"]
        target = namespace.get(entry)

        if not callable(target):
            what = "not defined" if target is None else f"defined but not callable ({type(target).__name__})"
            result.verdict = f"✗ no function named {entry}() — it is {what} in {spec['source']}"
            result.diagnosis = part_spec.get("missing_hint")
            report["parts"].append(result.to_dict())
            continue

        failed: list[str] = []
        for test in part_spec["tests"]:
            ok, value = _safe(lambda t=test: t["run"](target))
            if not ok:
                result.tests.append({"name": test["name"], "ok": False, "detail": f"your function raised {type(value).__name__}: {value}"})
                failed.append(test["name"])
                continue
            ok_flag, detail = value
            result.tests.append({"name": test["name"], "ok": bool(ok_flag), "detail": detail})
            if not ok_flag:
                failed.append(test["name"])

        result.passed = not failed
        if result.passed:
            result.verdict = "✓ all checks pass"
        else:
            result.verdict = f"✗ {len(failed)} of {len(result.tests)} checks failed"
            for entry_diag in part_spec.get("diagnose", []):
                matched, value = _safe(lambda d=entry_diag: d["when"](target))
                if matched and value:
                    result.diagnosis = entry_diag["hint"]
                    break

        report["parts"].append(result.to_dict())

    if figures:
        ok, value = _safe(lambda: checks.figures(namespace))
        if ok:
            report["figures"] = value
        else:
            report["figures"] = {"error": f"{type(value).__name__}: {value}"}

    return report


def solution_text(lesson_id: str) -> str | None:
    spec = LESSONS.get(lesson_id)
    if spec is None:
        return None
    path = ROOT / spec["solution"]
    return path.read_text() if path.exists() else None
