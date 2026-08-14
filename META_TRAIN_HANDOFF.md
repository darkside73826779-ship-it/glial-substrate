# Meta-Training Loop — Build & Result Handoff

**Date:** 2026-08-14
**Author of this build:** OpenClaw coding agent (Claude Opus) acting under `AGENT_ORDERS.md`
**Purpose of this doc:** hand the work back to the designing assistant (Claude) for interpretation and a ruling on flagged judgment calls. Rebecca is the human owner; she overrides everything.

---

## 0. TL;DR

- Built the meta-training loop from `DESIGN.md` open work #1 as a **new file only**: `scripts/meta_train.py`. No edits to `layer.py`, `model.py`, `test_invariants.py`, or the benchmarks.
- Invariant suite still prints `all invariants hold` (before and after the build).
- **Result across 3 seeds: a clean NEGATIVE.** Out-of-context recall did not clear chance consistently, and the meta-trained gates induced **catastrophic forgetting** (pretrain-pool accuracy collapsed ~0.75–0.88).
- Two of the weaknesses trace to **my own choices**, not the architecture: (a) a constant high neuromodulator during meta-exposure, and (b) no retention/forgetting pressure in the probe objective.
- **Three decisions requested** (Section 6). I did not proceed past them.

---

## 1. What was built

New file: `scripts/meta_train.py`. It implements the per-episode loop mandated by `AGENT_ORDERS.md §1`:

1. Sample a novel key→value mapping on the **held-out** pool.
2. Initialize episode-local `fast`/`trace`/`consol` as fresh zero tensors (never the module buffers).
3. **Exposure:** run sequences through `functional_forward(x, fast)`; accumulate `trace` via a differentiable mirror of `_accumulate_trace`; apply `functional_tick` every `tick_every` steps to update `fast`.
4. **Probe:** out-of-context queries (queried pair absent from the sequence) through `functional_forward` with the final `fast`.
5. Backprop probe loss to **GateNet + `log_plastic_lr` only** (`set_mode("meta")`).

Supporting behavior:
- Per-seed it pretrains a parameter-matched **baseline** (no substrate) and a **substrate** model from identical init, then meta-trains only the substrate's gates, then evaluates both on the real gradient-free runtime deployment path (`reset_substrate` → `runtime` mode → `runtime_exposure` with ticks → probes).
- `§2` conjunction is computed and printed; results saved to `meta_results.json`.
- New scalars logged to `metrics.jsonl` for the monitor: `probe_loss`, `gate_mean`, `gatenet_gradnorm` per episode (fields that already exist in the loop; plain floats).

### Gradient-firewall self-guard (per §1)
After every optimizer step, the trainer computes the GateNet gradient L2 norm and **raises `RuntimeError` if it is exactly 0** — catching the documented `.detach()` trap that the invariant suite cannot (it tests the path, not our use of it). This guard is permanent. It fired correctly throughout (nonzero norm every episode).

---

## 2. Results (full run: seeds 0,1,2; d_model=128, layers=4, d_ff=256, pretrain=3000, episodes=400, tick_every=4, meta_lr=1e-3, neuromod=1.0, plastic_lr=0.5, gate_bias=default)

chance = 1/16 = 0.0625

| seed | model     | oo-ctx before | oo-ctx after | oo Δ vs baseline | in-ctx held Δ | pretrain-pool forgetting |
|-----:|-----------|--------------:|-------------:|-----------------:|--------------:|-------------------------:|
| 0    | baseline  | 0.000 | 0.000 | +0.000 | +0.016 | +0.000 |
| 0    | substrate | 0.000 | 0.000 | +0.000 | −0.128 | **+0.858** |
| 1    | baseline  | 0.000 | 0.000 | +0.000 | −0.044 | +0.000 |
| 1    | substrate | 0.000 | 0.016 | +0.016 | +0.050 | **+0.752** |
| 2    | baseline  | 0.000 | 0.000 | +0.000 | −0.020 | +0.000 |
| 2    | substrate | 0.000 | 0.136 | +0.136 | +0.080 | **+0.877** |

Meta-phase telemetry: gate mean rose from ~0.30 to **~0.95–0.97**; GateNet grad-norm stayed healthy (~0.15–0.68); probe_loss did **not** meaningfully descend (~30–50 throughout); all values finite; no NaN/Inf.

Wall-clock: ~85–128 s per seed on a single GPU (torch 2.11+cu128).

---

## 3. §2 definition-of-done verdict

- **Clause 1 (effect): NOT MET.** Substrate out-of-context recall is 0.000 / 0.016 / 0.136 — not above chance across all seeds, and only marginally/noisily above baseline. Not a reliable effect.
- **Clause 2 (contract): MET.** `python scripts/test_invariants.py` → `all invariants hold`, suite unmodified.
- **Clause 3 (clean comparison): MET.** Baseline out-of-context stayed 0.000 across all seeds; baseline code path untouched by the new script.

Per `AGENT_ORDERS.md §5`, this is a valid reportable outcome: *meta-training at this scale/these settings does not lift out-of-context recall within these constraints.* The metric was not forced; stability bounds (`alpha`, `budget_frac`) were **not** touched.

---

## 4. Interpretation (the coding agent's read — for Claude to confirm or correct)

The loop trains, gradients flow, gates open — but they learn a **degenerate always-write policy** (gate mean → ~0.97). The single-episode out-of-context probe loss is minimized about as well by writing indiscriminately as by writing selectively, and **nothing in the objective penalizes destroying prior knowledge** — hence pretrain-pool accuracy collapses while out-of-context recall stays near chance. The occasional blip (seed 2 = 0.136) coincides with the worst forgetting and looks like aggressive-write noise, not durable recall.

Two contributors are the coding agent's own choices, not the architecture:
1. **Constant `neuromod = 1.0`** during meta-exposure removed the surprise-gating that the design (and the GateNet's negative bias init) uses to keep plasticity *off by default*. "Always write" became the easy minimizer.
2. **No retention term** in the probe objective — the loop is never asked to preserve anything, so it doesn't.

---

## 5. Documentation-compliance audit (honest, including gray areas)

**Clearly in keeping:**
- §0 verify-first done; §1 loop shape followed; backprop reaches only GateNet + `log_plastic_lr`; episode-local tensors, never buffers.
- No edits to `layer.py` / `model.py` / `test_invariants.py` / benchmarks. Contract green.
- §1 gradient-firewall self-guard implemented and permanent.
- §3: `alpha` / `budget_frac` untouched; ladder knobs exposed in order.
- §2/§5 honest negative reporting; metric not forced.
- MONITOR_RULES: monitor stayed a single static file, no backend, no control channel, no model hooks; new charts read only already-logged fields (explicitly permitted for the meta loop). NOTE: also fixed a pre-existing bug — `monitor.html` fetched `metrics.jsonl` relative to `/monitor/`, i.e. `/monitor/metrics.jsonl` (404); corrected to `../metrics.jsonl` per the README's serve-from-repo-root layout.

**Gray areas — flagged for a ruling:**

1. **Runtime `forward`-swap.** To run the whole transformer through the functional path without editing `model.py`, `meta_train.py` uses a context manager that **temporarily monkey-patches each `SubstrateLinear.forward`** (to call `functional_forward` with the episode-local `fast`) and restores it on exit. No file is modified and the invariant suite is green. But §1 lists "any change to `model.py` mode logic" as not-to-build; a strict reading could view runtime-swapping `forward` as changing forward *behavior*. Coding agent's position: this is outer-loop orchestration, and I-4/I-6 concern the runtime buffers (untouched) — but it is a judgment call. A cleaner alternative is a first-class functional forward path in `model.py`, which **is** a `model.py` edit and needs sign-off.

2. **Constant `neuromod = 1.0`.** Permitted by the tunable surface (the neuromod formula is explicitly tunable, I-5), but arguably against the design's intent (surprise-gated, off-by-default plasticity). Allowed by the letter, unfaithful to the spirit; likely the direct cause of the degenerate policy.

3. **Probe objective = out-of-context only (no retention term).** Faithful to §1's literal definition, but the result shows this objective is under-specified w.r.t. forgetting. Adding a retention/anti-forgetting term would extend the defined task, which per §2/§4 leans toward "ask the human first."

---

## 6. Decisions requested (I stopped here rather than proceed)

**D1 — The `forward`-swap.** Acceptable as outer-loop orchestration, or should the functional forward path be made first-class in `model.py` (a §-sign-off `model.py` edit)? If neither, propose the preferred mechanism for running the full model functionally.

**D2 — Neuromodulator.** For the next run, switch from constant `neuromod=1.0` to the **surprise-driven** signal (`SubstrateController.neuromod()` / its formula), so gates must learn *when* to write? (On the tunable surface; consistent with I-5.)

**D3 — Objective.** May the meta objective add a **retention / anti-forgetting term** (e.g., probe also scores previously-written or pretrain-pool pairs, penalizing their degradation), or must the probe stay strictly out-of-context as §1 literally defines it? This is the change most likely to fix the catastrophic forgetting, but it edits the *defined task*, so I want an explicit ruling.

**Also useful to hear:** whether a clean negative at this toy scale is itself an acceptable stopping point per §5, or whether D2/D3 should be attempted first before concluding.

---

## 7. Reproduction

```bash
# from repo root, in the isolated venv (.venv, torch 2.11+cu128, CUDA)
.\.venv\Scripts\python.exe scripts/test_invariants.py         # -> all invariants hold
.\.venv\Scripts\python.exe scripts/meta_train.py --fast       # smoke: 1 seed, ~seconds
.\.venv\Scripts\python.exe scripts/meta_train.py --seeds 0 1 2 # full run reported above
# live dashboard (README layout): serve repo root, open /monitor/monitor.html
.\.venv\Scripts\python.exe -m http.server 8137
```

Ladder knobs (use in §3 order; `alpha`/`budget_frac` deliberately NOT exposed):
`--gate_bias` (rung 1), `--neuromod` (rung 2 signal), `--plastic_lr` (rung 4),
plus `--episodes --episode_exposure --tick_every --meta_lr --exposure_steps`.

Artifacts: `scripts/meta_train.py`, `meta_results.json`, `metrics.jsonl`, monitor at `monitor/monitor.html`.

---

## 8. Files changed in this build

- **Added:** `scripts/meta_train.py` (the meta-training loop).
- **Added:** `META_TRAIN_HANDOFF.md` (this file).
- **Edited:** `monitor/monitor.html` — (a) fetch path bug fix `metrics.jsonl` → `../metrics.jsonl`; (b) two new observer-only meta-training charts reading already-logged fields.
- **Unchanged (verified):** `substrate/layer.py`, `substrate/model.py`, `substrate/tasks.py`, `substrate/metrics.py`, `scripts/test_invariants.py`, `scripts/run_experiment.py`, `DESIGN.md`, `AGENT_ORDERS.md`, `MONITOR_RULES.md`.
