"""Robust static server for the run monitor (observability rule).

Python's stock `python -m http.server` is single-threaded and blocks: one
stalled/half-open connection can wedge it and the dashboard goes dark even
though the run is fine. This uses ThreadingHTTPServer so the dashboard's 2s
polling can never starve. Still a pure read-only static server — no backend,
no write path, no experiment coupling (MONITOR_RULES.md respected).

Usage:  python monitor/serve.py [port]   # default 8137, serves repo root
"""
from __future__ import annotations

import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8137
root = str(Path(__file__).resolve().parents[1])  # repo root
handler = partial(SimpleHTTPRequestHandler, directory=root)
httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
print(f"serving {root} at http://127.0.0.1:{port}/  "
      f"(open /monitor/monitor.html)", flush=True)
httpd.serve_forever()
