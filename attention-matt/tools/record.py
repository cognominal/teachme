"""record.py — write the next learning record. `./bin/record`.

Learning records are untracked on purpose (they are notes about a person, not
course material), so a fresh clone has no `learning-records/` directory at all.
This script is the only thing that ever creates it: it makes the directory if it
is missing, scans the highest existing number, and writes the next one.

    ./bin/record "Attention is a weighted average"          # -> 0004-attention-is-a-weighted-average.md
    ./bin/record "Masks click" --evidence "graded 17/17" --implications "skip mask re-drill"
    ./bin/record --list                                     # what is already recorded
    ./bin/record "Draft" --dir /tmp/records                 # somewhere else

Format follows the teach skill's LEARNING-RECORD-FORMAT.md: a title, one to
three sentences, and the optional Status / Evidence / Implications sections.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "learning-records"
RECORD_RE = re.compile(r"^(\d+)-.*\.md$")


def slugify(text: str) -> str:
    """'Masks don't leak the future!' -> 'masks-dont-leak-the-future'."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    return re.sub(r"-{2,}", "-", slug)[:60].strip("-") or "record"


def records(directory: Path) -> list[tuple[int, Path]]:
    """Existing records as (number, path), sorted. A missing directory is fine."""
    if not directory.is_dir():
        return []
    found = []
    for path in directory.glob("*.md"):
        match = RECORD_RE.match(path.name)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found)


def next_number(directory: Path) -> int:
    return max((number for number, _ in records(directory)), default=0) + 1


def body(title: str, status: str | None, evidence: str | None, implications: str | None) -> str:
    lines = [f"# {title}", ""]
    if status:
        lines += [f"**Status**: {status}", ""]
    lines += [
        "<!-- 1-3 sentences: what was learned (or what prior knowledge was established), "
        "and why it matters for future sessions. -->",
        "",
    ]
    if evidence:
        lines += ["**Evidence**: " + evidence, ""]
    if implications:
        lines += ["**Implications**: " + implications, ""]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Write the next learning record, creating learning-records/ if needed.")
    parser.add_argument("title", nargs="?", help="short title of what was learned")
    parser.add_argument("--slug", help="file slug (default: derived from the title)")
    parser.add_argument("--status", help='e.g. "active", or "superseded by LR-0002"')
    parser.add_argument("--evidence", help="how the learner demonstrated it (a question, an exercise, prior experience)")
    parser.add_argument("--implications", help="what this unlocks or rules out next")
    parser.add_argument("--dir", default=str(DEFAULT_DIR), help=f"where records live (default: {DEFAULT_DIR})")
    parser.add_argument("--list", action="store_true", help="list the records and stop")
    args = parser.parse_args(argv)

    directory = Path(args.dir).expanduser()

    if args.list:
        existing = records(directory)
        if not existing:
            print(f"no records in {directory} (it {'exists' if directory.is_dir() else 'does not exist'})")
            return 0
        for number, path in existing:
            first_line = path.read_text(encoding="utf-8").splitlines()[0:1]
            heading = first_line[0].lstrip("# ").strip() if first_line else ""
            print(f"  {number:04d}  {path.name}  {heading}")
        return 0

    if not args.title:
        parser.error("give a title, or --list")

    existed = directory.is_dir()
    directory.mkdir(parents=True, exist_ok=True)
    if not existed:
        print(f"created {directory}{'' if directory == DEFAULT_DIR else '/'}")

    number = next_number(directory)
    slug = args.slug or slugify(args.title)
    path = directory / f"{number:04d}-{slug}.md"
    for _, other in records(directory):
        if other.name.endswith(f"-{slug}.md"):
            print(f"note: {other.name} already covers this slug — writing a new record anyway (records are chronological)")

    path.write_text(body(args.title, args.status, args.evidence, args.implications), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
