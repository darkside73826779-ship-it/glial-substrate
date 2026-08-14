# MONITOR_RULES.md — Observer Boundary for the Coding Agent

**Scope:** `monitor/monitor.html`, `substrate/metrics.py`, and any future observability work.
**Authority:** supplements `AGENT_ORDERS.md`. Same chain: DESIGN.md invariants > AGENT_ORDERS > this file > agent judgment. Human overrides all.

## The one rule everything else derives from

**The monitor is a read-only observer. There is no channel from the UI back into a running experiment, and there never will be one.**

The architecture enforcing this: training scripts write append-only JSONL via `MetricsLog`; the dashboard polls that file over HTTP; neither side imports, signals, or knows about the other beyond the file format. This is deliberate — it makes observer-effect contamination structurally impossible, which is what lets Rebecca watch runs without the watching becoming a variable.

## The deletion test (run it after any monitor work)

Delete `monitor/` and `metrics.jsonl`, then run the invariant suite and a `--fast` experiment. **Both must behave identically to before your edits** (suite green, experiment completes, results within seed noise). If training behavior changes in any way when the monitor's files are absent, you have created a dependency across the boundary. Revert.

## Forbidden — do not build these, even if asked to "improve" the monitor

1. **Any control that affects a running experiment.** No pause/resume, no stop button, no live hyperparameter sliders, no "restart with tweaked config," no early-stop-when-flat automation. Live-adjusting an experiment invalidates the run even when invariants stay green — the §2 definition of done assumes fixed conditions per run. If Rebecca wants a control panel someday, that is a launcher for *new* runs, built as a separate tool, never wired into a live one.
2. **Hooks or instrumentation added to model code for the UI's benefit.** Metrics are computed inside the existing tick and training loops, where the values already exist, and passed to `MetricsLog` as plain floats. Never register forward/backward hooks, never add tensor operations to `forward()` or `tick()` to feed the dashboard, and never retain tensor or graph references for logging — during meta-training a retained reference can silently keep autograd graphs alive and change memory behavior. `.item()` at the existing computation site, then plain Python numbers only.
3. **The logger acquiring opinions.** `MetricsLog` stays append-only, dependency-free, and dumb. No locks, no truncation, no rotation-in-place (rotate by starting a new file if ever needed), no reading its own file, no callbacks. The writer appends; the reader polls; JSONL's line-atomicity is the whole concurrency story.
4. **The dashboard gaining a backend.** `monitor.html` stays a single self-contained static file with zero dependencies, served by `python -m http.server` or any static server. No websocket server, no Flask app, no process that could conceivably grow a write path. If polling feels inefficient, it is — and it is also the point.
5. **Health flags becoming actions.** The dashboard's warning badges (gates collapsed, gates saturated, NaN detected) are observations for the human, mapped to the documented failure modes in DESIGN.md. They must never trigger anything.

## Allowed monitor work

- Visual/layout improvements, more charts, per-layer breakdowns — provided the data comes from fields already being logged, or from new fields added *at existing logging call sites* as plain floats.
- Logging new scalars from the meta-training loop when you build it (probe accuracy per episode, GateNet gradient norm, episode wall-clock). Same pattern: values that already exist in the loop, `.item()`, pass to `mlog.log()`.
- A `--log_every N` CLI flag to thin high-frequency logging if file size becomes a problem at scale.
- Historical run comparison (loading multiple JSONL files side by side) — reading more files is always fine.

## Why this is strict

Every one of the forbidden items is a natural, helpful-seeming next step, which is exactly why they are written down. A pause button is convenient; it is also a write channel. A hook is the easy way to get per-layer stats; it is also code in the measured path. The experiment's credibility rests on the claim that nothing observed the substrate except the substrate — keep it literally true.
