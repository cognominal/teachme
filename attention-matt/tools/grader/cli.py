"""cli.py — the terminal front end: `./bin/grade 0001`.

Same checks, same diagnoses as the in-page runner. Prints a readable report and
exits non-zero when something fails, so it composes with a shell prompt.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402

OK = "\033[32m"
BAD = "\033[31m"
DIM = "\033[2m"
BOLD = "\033[1m"
OFF = "\033[0m"


def use_colour(stream) -> bool:
    return hasattr(stream, "isatty") and stream.isatty()


_TAGS = __import__("re").compile(r"<[^>]+>")
_ENTITIES = {"&lt;": "<", "&gt;": ">", "&amp;": "&", "&nbsp;": " "}


def plain(html: str) -> str:
    """The hints are written once, in the page's HTML; the terminal gets the text."""
    text = html
    for entity, char in _ENTITIES.items():
        text = text.replace(entity, char)
    return _TAGS.sub("", text)


def render(report: dict, colour: bool = True) -> str:
    def ok(text):  # noqa: ANN001
        return f"{OK}{text}{OFF}" if colour else text

    def bad(text):  # noqa: ANN001
        return f"{BAD}{text}{OFF}" if colour else text

    def dim(text):  # noqa: ANN001
        return f"{DIM}{text}{OFF}" if colour else text

    def bold(text):  # noqa: ANN001
        return f"{BOLD}{text}{OFF}" if colour else text

    out: list[str] = []
    out.append(bold(f"Lesson {report['lesson']} · {report.get('title', '')}"))
    out.append(dim(f"grading {report.get('source', '')}"))

    error = report.get("error")
    if error:
        kind = error["kind"]
        if kind == "missing":
            out.append(bad(f"✗ {error['message']}"))
        elif kind == "syntax":
            out.append(bad(f"✗ Python could not compile your file: {error['message']}"))
            if error.get("line"):
                out.append(dim(f"    line {error['line']}: {error.get('text') or ''}"))
        else:
            out.append(bad("✗ your file raised while loading:"))
            out.append(dim(error["message"]))
        return "\n".join(out)

    for part in report["parts"]:
        out.append("")
        out.append(bold(part["title"]))
        if part.get("verdict"):
            out.append("  " + (ok(part["verdict"]) if part["passed"] else bad(part["verdict"])))
        for test in part["tests"]:
            mark = ok("✓") if test["ok"] else bad("✗")
            detail = f" {dim('— ' + test['detail'])}" if test["detail"] else ""
            out.append(f"  {mark} {test['name']}{detail}")
        if part.get("diagnosis"):
            out.append(f"  {bad('most likely cause')} {plain(part['diagnosis'])}")

    output = report.get("output")
    if output and (output.get("stdout") or output.get("stderr")):
        out.append("")
        out.append(bold("Your file's own output"))
        out.append(dim(f"  $ ./.venv/bin/python {report.get('source', '')}"))
        for line in output["stdout"].rstrip("\n").splitlines():
            out.append("  " + line)
        if not output["stdout"].strip() and not output["stderr"]:
            out.append(dim("  (it printed nothing)"))
        if output["stderr"]:
            if output.get("timed_out"):
                out.append(bad("  ✗ it did not finish within 20 seconds"))
            else:
                out.append(bad(f"  ✗ it stopped with exit code {output.get('returncode')}"))
            for line in output["stderr"].rstrip("\n").splitlines():
                out.append(dim("  " + line))
        if output.get("truncated"):
            out.append(dim("  (stdout truncated)"))

    passed = [p for p in report["parts"] if p["passed"]]
    out.append("")
    if report["parts"] and len(passed) == len(report["parts"]):
        out.append(ok("all parts pass ✓" + next_hint(report["lesson"])))
    else:
        out.append(bad(f"{len(report['parts']) - len(passed)} of {len(report['parts'])} parts still failing"))
    out.append(dim("(the same verdicts appear on the lesson page when ./bin/grader is running)"))
    return "\n".join(out)


def next_hint(lesson: str) -> str:
    """Point at the next lesson written so far, without naming one that is not there."""
    order = list(core.LESSONS)
    if lesson in order:
        following = order[order.index(lesson) + 1:]
        if following:
            return f"  — next: lesson {following[0]}"
    return "  — that is the last lesson written so far"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grade this workspace's Python exercises.")
    parser.add_argument("lesson", nargs="?", default="0001", help="lesson id, e.g. 0001 (default: 0001)")
    parser.add_argument("--json", action="store_true", help="print the raw report")
    parser.add_argument("--file", type=Path, default=None, help="grade a different file with this lesson's checks")
    parser.add_argument("--show-solution", action="store_true", help="print a reference solution instead of grading")
    parser.add_argument("--no-colour", action="store_true")
    args = parser.parse_args(argv)

    if args.show_solution:
        text = core.solution_text(args.lesson)
        if text is None:
            print(f"no solution on file for lesson {args.lesson}", file=sys.stderr)
            return 2
        print(text)
        return 0

    report = core.grade(args.lesson, source=args.file)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render(report, colour=not args.no_colour and use_colour(sys.stdout)))
    return 0 if report.get("error") is None and all(p["passed"] for p in report["parts"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
