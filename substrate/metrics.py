"""Metrics logger: append-only JSONL, no dependencies, no opinions.

The contract with the dashboard (monitor.html):
  - one JSON object per line, always containing "t" (unix time) and "step"
  - everything else is free-form numeric fields
  - the run's identity line has "event": "run_start" with config metadata
  - phase markers have "event": "phase" with "name"

Training code writes; the dashboard polls the file. Neither knows the
other exists beyond this file format, so the monitor can never break a
run (see AGENT_ORDERS.md cleanup rules — this module is deliberately
outside the invariant surface).
"""

from __future__ import annotations

import json
import time
from pathlib import Path


class MetricsLog:
    def __init__(self, path: str | Path = "metrics.jsonl", run_name: str = "run",
                 config: dict | None = None, append: bool = False):
        self.path = Path(path)
        if not append and self.path.exists():
            self.path.unlink()
        self.step = 0
        self._write({"event": "run_start", "run": run_name,
                     "config": config or {}})

    def _write(self, obj: dict) -> None:
        obj.setdefault("t", time.time())
        obj.setdefault("step", self.step)
        with self.path.open("a") as f:
            f.write(json.dumps(obj) + "\n")

    def phase(self, name: str) -> None:
        self._write({"event": "phase", "name": name})

    def log(self, **fields) -> None:
        self.step += 1
        clean = {}
        for k, v in fields.items():
            if hasattr(v, "item"):
                v = v.item()
            if isinstance(v, (int, float, str, bool)):
                clean[k] = v
        self._write(clean)
