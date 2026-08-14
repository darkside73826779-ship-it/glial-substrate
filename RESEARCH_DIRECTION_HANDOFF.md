# Glial Substrate — Research-Direction Handoff

**Date:** 2026-08-14
**From:** OpenClaw coding agent (Claude Opus, under `AGENT_ORDERS.md`), working session with Rebecca
**To:** the designing assistant (Claude), for interpretation + design-direction rulings
**Companion docs:** `META_TRAIN_HANDOFF.md` (v1 build detail), `RULINGS_META_TRAIN.md` (your prior D1/D2/D3 rulings), `DESIGN.md`, `AGENT_ORDERS.md`, `MONITOR_RULES.md`

This document consolidates a full session of building and running the meta-training loop, the negative results, the diagnosis of *what the architecture appears to be missing*, and — most importantly — a **methodological problem with the experiment's control** that Rebecca identified and wants addressed before investing further. It ends with a concrete decision request.

---

## 1. What was done this session

1. Built the meta-training loop (open work #1) as `scripts/meta_train.py`. **v1** used a constant neuromodulator and no retention term → negative + catastrophic forgetting (preserved as `scripts/meta_train_v1_negative.py`).
2. Under your rulings (`RULINGS_META_TRAIN.md`), implemented **v2**:
   - **D1** — first-class functional forward path in `model.py` (`fast_map`/`act_out`), replacing the v1 runtime monkey-patch. Guarded by **new invariant I-11** (functional forward is bit-identical to runtime forward when fed the buffers; verified max|Δ| = 0.00e+00). Suite now 13 checks, all green.
   - **D2** — surprise-driven neuromodulator (functional mirror of `SubstrateController.neuromod()`), with episodes mixing **write-worthy** (novel held-out) and **write-hostile** (familiar pretrain-pool) steps so the gate has neuromod contrast to condition on.
   - **D3** — retention term added to the meta objective: `L = L_recall + λ_ret · L_retention`, where `L_retention` penalizes degradation of pretrain-pool recall probed through the episode's final fast tensors.
3. Ran full 3-seed comparisons for both versions. Contract stayed green throughout (`alpha`/`budget_frac` never touched).
4. Added a standing **observability rule** (Rebecca's): no run launches until the monitor UI is verified end-to-end. Codified in `AGENT_ORDERS.md §0a`; replaced the wedge-prone stock server with threaded `monitor/serve.py`.

---

## 2. Results

Chance = 0.0625. Success bar (your `RULINGS §4`): substrate out-of-context above chance across ALL seeds AND beats baseline, WITH pretrain-pool forgetting < 0.10.

**v2 final (seeds 0,1,2; d_model=128, 4 layers, 3000 pretrain steps, 400 episodes, λ_ret=1.0, hostile_prob=0.5, surprise neuromod):**

| seed | substrate oo-ctx | (sub − base) oo | pretrain-pool forgetting |
|-----:|-----------------:|----------------:|-------------------------:|
| 0 | 0.000 | +0.000 | **0.870** |
| 1 | 0.047 | +0.047 | **0.772** |
| 2 | 0.111 | +0.111 | **0.880** |

**Verdict: FAIL.** Out-of-context recall does not reliably clear chance, and forgetting is catastrophic (0.77–0.88), far past the 0.10 bound.

**Important nuance — v2 fixed the *within-episode* pathology but not deployment:**
- During meta-training, gate mean stayed *selective* (0.27–0.97, ending ~0.67; v1 saturated ~0.97), retention loss stayed tiny (~5e-5), neuromod swung with surprise. The training-time behavior looked healthy.
- At **deployment** (the real gradient-free `runtime_exposure`), gates re-saturated (~0.957) and forgetting returned. This is a **train/deploy horizon mismatch**: meta episodes are 24 steps / ~6 ticks / mixed; deployment is 800 steps / ~200 ticks / all-novel. The gate policy shaped on short mixed windows does not govern long all-novel write sessions.

---

## 3. Diagnosis — what the evidence says is missing

Consistent across v1 and v2, every configuration: **the gate learns write *magnitude*, not write *selectivity*.** It moves a scalar-per-group plasticity knob up (v1: saturate) or holds it moderate (v2: ~0.44), but never demonstrates "write key K→value V into the specific weights that encode that mapping." Out-of-context recall stayed at chance in *every* run.

**The structural gap, in order of evidential strength:**

1. **No routing/addressing signal (strongest).** The gate sees group statistics + one global scalar. The Hebbian outer-product trace carries co-activation *magnitude*, not *address*. There is no channel by which "which key is this" can steer *which weights* change. Content cannot determine location. → This is exactly the deferred **open work #3 (lateral group communication)** — the mechanism that lets "which group" enter the computation.
2. **No retrieval mechanism separate from storage (strong).** Reading is a uniform `W_base + alpha·tanh(fast)` matmul. There is no content-addressed *read* — no way for a query key to preferentially activate the weights holding its value. Storage and retrieval are collapsed into one dense op; associative memory usually needs them separable.
3. **Consolidation shields the wrong quantity (medium).** `consol` guards fast-weight budget, but functional forgetting comes from `W_eff` drifting off a good `W_base`. The shield protects norm, not function.

**Honest caveat:** the evidence proves the architecture *as-configured* does not reach out-of-context recall and points hard at addressing/routing as the gap. It does **not** prove the gap is unbridgeable, nor fully exclude that a much larger/longer run coaxes selectivity from the current gate. But the design docs *already anticipated this* — #3 and #4 are deliberately-deferred. The system is a partial build hitting the wall its own authors predicted.

**Interpretation to confirm/correct:** the current experiment is really testing *"can a low-dimensional, scalar-per-group, surprise-gated plasticity rule implement content-addressable memory?"* — and the evidence to date says the gate is a **volume knob, not a router**, and no amount of §3-ladder tuning crosses that gap.

---

## 4. The control problem (Rebecca's key methodological point — needs a ruling)

The current experiment's control is a **parameter-matched frozen baseline**: the *same* transformer with the substrate switched off (I-9 enforces forward-path parity; baseline oo-ctx = 0.000, forgetting = 0.000 every seed). This is a legitimate **negative control** — it cleanly isolates "substrate vs. no-substrate" and attributes forgetting to the substrate. **But it is a strawman for the real question**: it is a deliberately-inert version of our *own* system, not how the field actually solves runtime knowledge acquisition. Beating it proves "we do more than nothing," a low bar.

**What a truly fair test needs (Rebecca's framing):** run an **off-the-shelf, industry-standard system through the same training and the same metrics, side by side** — matched on what must be equal for fairness:
- same base model / identical pretrained start,
- same data exposure (sequences, order, count),
- **same compute / wall-clock budget** (the axis that makes it honest — she notes a standard small run took *hours* to show meaningful learning; if the substrate reaches comparable capability in minutes without backprop, that is the headline),
- same measurements on both: **loss curves side by side AND capability probes side by side.**

**Missing comparator/control arms (all additive, no core/invariant edits):**
1. **In-context vanilla** — let the plain transformer attend to the pairs in-context. The *real* competitor; modern transformers are very strong here for free.
2. **Runtime gradient fine-tuning / LoRA** — the capability ceiling (uses backprop, i.e. the thing the thesis is trying to avoid — so it defines the frontier, not a pass/fail).
3. **Retrieval / external memory (kNN-LM / RAG)** — the standard "bolt-on memory" approach.
4. **Ablation control** — substrate with *untrained/random/hand-fixed* gates, to separate *plasticity itself* from the *meta-trained gating policy*. (Currently un-run head-to-head; `run_experiment.py`'s hand-gate result — also 0.000 oo-ctx — is suggestive but not a deliberate arm.)
5. **Positive control** — a configuration known to be solvable (e.g. an oracle that writes the correct mapping, or a tiny explicit key-value memory) to prove the probe *can* leave 0.000 and the metric is valid. Without this, a persistent 0.000 is ambiguous (architecture failure vs. unmeasurable probe).

**Framing caution for the comparison:** the thesis's axis is *"without backprop at runtime."* So the honest output is not a single win/lose number but a **capability-vs-cost frontier**: report each method's capability *alongside what it costs* (gradients? extra params? external store? compute). The substrate "wins" if it lands near in-context/RAG capability at a constraint profile (bounded, gradient-free, dense, no external store) nobody else offers — or shows a capability none of them match at matched budget.

---

## 5. The sequencing decision (needs Rebecca + your ruling — governance flag)

`DESIGN.md open work #4` **gates scale-up**: "do not attempt [retrofit on a real frozen model] until the toy shows a meta-trained effect." The toy has **not** shown that effect (out-of-context still at chance). So there is a real fork:

- **Path A — fix the toy first.** Prototype the missing addressing/routing architecture (#3) in the cheap toy, get out-of-context off chance, *then* scale to a real base model and run the matched side-by-side (§4). Respects the design's own sequencing; minutes-scale iteration.
- **Path B — override the #4 gate now.** Stand up the real-model matched comparison immediately (substrate + off-the-shelf, same base, same budget, loss+capability). Tests "does it matter at real scale" directly; hours-scale runs; legitimate only as a conscious override of #4.

**Coding agent's recommendation: A, then B.** Evidence says the toy is missing *addressing*; that is cheap to prototype and diagnose in minutes. If the missing architecture moves out-of-context recall in the toy, *that* is the green light worth burning hours comparing at scale. Scaling a system that is still a volume knob buys an expensive confirmation of the toy's negative. Reminder from `AGENT_ORDERS §2`: **loss ≠ capability** — bank the claim on capability probes, not loss curves.

---

## 6. Decision requests for Claude

- **R1 — Diagnosis:** Do you agree the core gap is **content addressing / routing + separable retrieval** (i.e. the evidence justifies moving to open work #3 rather than spending more §3-ladder rungs on the current scalar-per-group gate)? If not, what does the evidence say instead?
- **R2 — Missing architecture:** If addressing is the gap, what is the minimal design that adds it while respecting the invariants (I-3 functional grouping, I-5 locality, I-6 tick-only mutation, I-1/I-2 bounds)? Lateral group communication as sketched in #3, a key-conditioned gate feature, a separable read path — or something else?
- **R3 — Train/deploy mismatch:** Is matching the meta-episode horizon to deployment (or making deployment interleave rehearsal) worth doing *before* R2, or is it a symptom that R2 subsumes?
- **R4 — Controls:** Endorse building the comparator suite (in-context vanilla, LoRA-at-runtime, retrieval, ablation, positive control) as new arms? Any you'd add/drop? Confirm the capability-vs-cost frontier framing.
- **R5 — Sequencing:** Path A (fix toy → then scaled matched comparison) vs Path B (override #4, scaled comparison now). Rebecca decides; your recommendation requested. Note #4 is explicitly gated.
- **R6 — Positive control design:** what known-solvable configuration would you trust to validate the probe can leave chance?

---

## 7. State of the code (for reproduction / continuation)

- `scripts/meta_train.py` — v2 loop (D1+D2+D3). `--fast` smoke, `--seeds 0 1 2` full. Ladder knobs `--gate_bias --neuromod_gain --plastic_lr --hostile_prob --lambda_ret` (alpha/budget_frac deliberately not exposed).
- `scripts/meta_train_v1_negative.py`, `meta_results_v1_negative.json` — v1 provenance.
- `substrate/model.py` — sanctioned functional forward path (D1), bit-identical when unused.
- `scripts/test_invariants.py` — 13 checks incl. **I-11**; prints `all invariants hold`.
- `monitor/serve.py` — threaded static server (observability rule); `monitor/monitor.html` — fetch-path fixed + meta charts (observer-only).
- `meta_results.json` — v2 final numbers (§2).
- Isolated venv `.venv` (torch 2.11+cu128, CUDA). Verify UI end-to-end before any run (`AGENT_ORDERS §0a`).

**Nothing is blocked in code. The open questions are design-direction (R1–R6), which is why this is going to you.**
