"""check_lessons.py — syntax-check every inlined <script> in every lesson.

A lesson page is HTML with JavaScript inlined, so a stray quote is a bug that
only shows up when a browser loads the page. This catches it in the terminal
instead: extract each inline script, hand it to a JS parser, report line numbers
mapped back to the HTML file.

    ./.venv/bin/python tools/check_lessons.py            # all lessons
    ./.venv/bin/python tools/check_lessons.py lessons/0001-*.html
"""

from __future__ import annotations

import glob
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_RE = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", re.DOTALL | re.IGNORECASE)


def find_js_runtime() -> list[str] | None:
    for candidate in (["node", "--check"], ["bun", "build", "--target=browser"]):
        if shutil.which(candidate[0]):
            return candidate
    return None


def check(path: Path, runtime: list[str]) -> list[str]:
    html = path.read_text()
    problems: list[str] = []
    for match in SCRIPT_RE.finditer(html):
        code = match.group(1)
        if not code.strip():
            continue
        start_line = html[: match.start(1)].count("\n") + 1
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
            handle.write(code)
            temp = Path(handle.name)
        if runtime[0] == "node":
            proc = subprocess.run([*runtime, str(temp)], capture_output=True, text=True)
        else:  # bun build: parses the file, writes nothing useful
            proc = subprocess.run([*runtime, str(temp), "--outfile=/dev/null"], capture_output=True, text=True)
        if proc.returncode != 0:
            output = (proc.stderr or proc.stdout).strip().splitlines()
            detail = "\n      ".join(output[:6])
            problems.append(f"{path.name}: inline script starting at HTML line {start_line}\n      {detail}")
        temp.unlink(missing_ok=True)
    return problems


def main(argv: list[str]) -> int:
    patterns = argv or [str(ROOT / "lessons" / "*.html"), str(ROOT / "assets" / "*.js")]
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(pattern))

    runtime = find_js_runtime()
    if runtime is None:
        print("no node or bun on PATH — cannot syntax-check", file=sys.stderr)
        return 2

    problems: list[str] = []
    for path in paths:
        if path.suffix == ".js":
            proc = subprocess.run([*runtime, str(path)], capture_output=True, text=True)
            if proc.returncode != 0:
                problems.append(f"{path.name}\n      " + "\n      ".join((proc.stderr or proc.stdout).strip().splitlines()[:6]))
        elif path.suffix == ".html":
            problems.extend(check(path, runtime))

    if problems:
        print(f"{len(problems)} problem(s):\n")
        for problem in problems:
            print(f"  ✗ {problem}\n")
        return 1
    print(f"✓ {len(paths)} file(s) parse cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
