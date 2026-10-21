"""serve.py — the local front end: `./bin/grader`.

Serves this workspace over http (so lesson pages can fetch their own verdicts
from a same-origin endpoint) and exposes:

    GET /health                 -> {"ok": true, "lessons": [...]}
    GET /grade?lesson=0001      -> the report `bin/grade` prints, as JSON

Binds 127.0.0.1 only. It never executes code that arrived over the network: the
only file graded is the one the lesson names under exercises/.
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402

DEFAULT_PORT = 8757


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
        super().do_GET()

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
            if url_path.startswith("/grade") or url_path.startswith("/health"):
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
    print(f"  health : {base}/health")
    print(f"  lesson : {base}/lessons/0001-attention-as-a-weighted-average.html")
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
