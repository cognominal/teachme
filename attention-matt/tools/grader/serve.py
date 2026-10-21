"""serve.py — the local front end: `./bin/grader`.

Serves this workspace over http (so lesson pages can fetch their own verdicts
from a same-origin endpoint) and exposes:

    GET /                       -> the lesson index: every page in lessons/, linked
    GET /health                 -> {"ok": true, "lessons": [...]}
    GET /grade?lesson=0001      -> the report `bin/grade` prints, as JSON

The index is built from the directory on every request, so a new lesson page
shows up there without restarting (unlike /health, which reads the registry that
Python imported once).

Binds 127.0.0.1 only. It never executes code that arrived over the network: the
only file graded is the one the lesson names under exercises/.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from functools import partial
from html import escape
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402

DEFAULT_PORT = 8757
TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.DOTALL | re.IGNORECASE)
LESSON_NUMBER_RE = re.compile(r"^\s*Lesson\s+\d+\s*[—–-]\s*", re.IGNORECASE)


def lesson_pages() -> list[dict[str, str]]:
    """Every page in lessons/, in numbering order, labelled with its own <title>.

    The page names itself; this only strips the "Lesson NN — " prefix, because the
    id is shown in its own column.
    """
    pages: list[dict[str, str]] = []
    for path in sorted((core.ROOT / "lessons").glob("*.html")):
        match = TITLE_RE.search(path.read_text(encoding="utf-8", errors="replace"))
        title = (match.group(1) if match else path.stem).strip()
        pages.append(
            {
                "id": path.name.split("-", 1)[0],
                "title": LESSON_NUMBER_RE.sub("", title) or title,
                "href": f"/lessons/{path.name}",
            }
        )
    return pages


def index_html() -> str:
    """The page served at `/`: the lessons, by name, as links."""
    rows = "\n".join(
        f'      <tr><td><code>{escape(page["id"])}</code></td>'
        f'<td><a href="{escape(page["href"])}">{escape(page["title"])}</a></td></tr>'
        for page in lesson_pages()
    ) or '      <tr><td colspan="2">no pages in <code>lessons/</code></td></tr>'
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>attention-matt — lessons</title>
<link rel="stylesheet" href="/assets/lesson.css">
</head>
<body>

<p class="kicker">attention-matt · served by <code>./bin/grader</code></p>
<h1>Lessons</h1>
<p class="subtitle">Each page is self-contained, and its exercise panel grades the file it names
through this server — so open lessons from here rather than from disk.</p>

<table>
  <thead><tr><th>#</th><th>Lesson</th></tr></thead>
  <tbody>
{rows}
  </tbody>
</table>

</body>
</html>
"""



class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory: str | None = None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    # -- shared headers -----------------------------------------------------
    def end_headers(self):
        # `*` also covers the `null` Origin that a file:// lesson page sends.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    # -- routes -------------------------------------------------------------
    def do_GET(self):  # noqa: N802
        url = urlparse(self.path)
        if url.path in ("/health", "/grade"):
            self._json_route(url)
            return
        if url.path in ("/", "/index.html"):
            self._index_route()
            return
        super().do_GET()

    def _index_route(self):
        """The lesson index, rebuilt from `lessons/` on every request."""
        body = index_html().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json_route(self, url):  # noqa: ANN001
        query = parse_qs(url.query)
        if url.path == "/health":
            payload = {"ok": True, "lessons": sorted(core.LESSONS)}
        else:
            lesson = (query.get("lesson") or ["0001"])[0]
            payload = core.grade(lesson)
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # noqa: ANN001, ANN002
        if url_path := getattr(self, "path", ""):
            if url_path in ("/", "/index.html") or url_path.startswith(("/grade", "/health")):
                sys.stderr.write(f"  {self.command} {url_path} → {fmt % args}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve this teaching workspace and grade on request.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)

    handler = partial(Handler, directory=str(core.ROOT))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    base = f"http://{args.host}:{args.port}"
    print(f"grader listening on {base}")
    print(f"  lessons: {base}/                      ← every page, linked")
    print(f"  health : {base}/health")
    print("  ctrl-c to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
