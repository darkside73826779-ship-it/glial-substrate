# AGENT_ORDERS.md — Post-Build Instructions for the Coding Agent

**Audience:** the coding agent (Opus via OpenClaw) working on this project after the initial extraction and build.
**Authority:** these orders supplement `DESIGN.md`. Where they appear to conflict, `DESIGN.md` invariants win, then these orders, then your own judgment. The human (Rebecca) overrides everything.

---

## 0a. Observability gate (Rebecca's standing rule, 2026-08-14)

**No experiment run may be launched until the monitor UI is verified working end-to-end, in the same turn, immediately before the run.** The human observes runs live; a dark UI means she is blind, which is not acceptable for "further testing."

Verification is not "the port is listening." It is, mechanically, all of:
1. `GET /monitor/monitor.html` returns 200 with nonzero length.
2. `GET /monitor/../metrics.jsonl` (the exact path the page fetches) returns 200.
3. The served HTML contains the expected charts for the run being launched (e.g. the meta panel for a meta run) and the correct `fetch("../metrics.jsonl")` path.

Use the threaded server `monitor/serve.py` (NOT stock `python -m http.server`, which is single-threaded and wedges on a half-open connection — the failure that blinded the 2026-08-14 v2 run mid-flight). Bind IPv4 `127.0.0.1` so `localhost` resolves. If any check fails, fix and re-verify BEFORE launching. State "UI verified" with the check results in the same turn as the launch.

## 0. Verify before touching

Before any edit, in this exact order:

1. Extract the tarball into a clean working directory.
2. Run `python scripts/test_invariants.py`. It must print `all invariants hold`.
3. Run `python scripts/run_experiment.py --fast`. It must complete without error and write `results.json`.

If either fails in the clean, unedited state, **stop and report**. That is an environment problem, not a code problem. Do not "fix" core files to make a broken environment pass.

Record the clean-state baseline numbers from `results.json`. You will need them for the definition of done (§2).

---

## 1. The task: build the meta-training loop

Your first and only initial task is `DESIGN.md` open work #1: an outer training loop that teaches the plasticity gates what to write.

**Shape of the loop (per episode):**
1. Sample a novel key→value mapping on the held-out pool.
2. Initialize episode-local `fast`, `trace`, `consol` tensors (zeros, correct shapes). These are ordinary tensors passed by value — **never the module buffers**.
3. Exposure phase: run sequences through `functional_forward(x, fast)`, compute traces functionally (mirror the logic of `_accumulate_trace`, but on local tensors, differentiably where needed), and apply `functional_tick(fast, trace, consol, neuromod)` at intervals to produce updated fast tensors.
4. Probe phase: out-of-context queries (pairs not shown in the probe sequence) through `functional_forward` with the final fast tensors.
5. Backpropagate probe loss to `GateNet` parameters and `log_plastic_lr` **only**. Step the optimizer on those parameters only.

**What you are NOT building:** any change to `layer.py`, `model.py` mode logic, the invariant tests, or the benchmark definitions. The differentiable path you need already exists and is verified (invariant I-4d). You are writing an outer loop around proven machinery. If you believe `layer.py` must change to complete the task, stop and report why — do not edit it speculatively.

**Expected resource behavior:** backprop through unrolled ticks holds the whole episode's graph in memory. If you hit memory limits, the correct responses are, in order: shorter episodes, fewer inner ticks per episode, smaller batch, gradient checkpointing. The **forbidden** response is detaching tensors anywhere on the functional path — a single misplaced `.detach()` silently severs gradients to GateNet and produces a trainer that runs flawlessly and learns nothing. The invariant suite will NOT catch this (it tests the path, not your use of it), so you must guard it yourself: after your first training step, assert that at least one `GateNet` parameter has a nonzero gradient, and keep that assertion in the trainer permanently.

---

## 2. Definition of done — a conjunction, all three required

The task is complete only when ALL of the following hold simultaneously:

1. **Effect:** meta-trained out-of-context recall on held-out pairs is above chance (chance = 1/16 ≈ 0.0625) by a margin that survives 3 different seeds, and exceeds the frozen baseline's number on the same probes.
2. **Contract:** `python scripts/test_invariants.py` prints `all invariants hold`, unmodified.
3. **Clean comparison:** the baseline model's numbers (pretrain accuracy, in-context accuracy, out-of-context accuracy) are unchanged from the clean-state baseline you recorded in §0, within seed noise. If baseline numbers drift, your edits touched shared code paths and the experimental comparison is contaminated — this clause is the one most easily broken silently, and skipping it invalidates everything.

Loss going down is not done. Gates looking active is not done. Do not substitute proxy metrics for the conjunction.

---

## 3. If training stalls — fix order is mandatory

Work down this list in order. Do not skip ahead.

1. Raise the `GateNet` final-layer bias init (gates may start too closed; they are initialized mostly-off by design).
2. Raise neuromodulator gain `k` in `SubstrateController` / the episode's neuromod signal.
3. Lengthen exposure (more sequences or more ticks per episode).
4. Adjust `plastic_lr` (via its init value).
5. Tune trace/fast decay rates.

**Last, and only with explicit human sign-off: `alpha` and `budget_frac`.** These are the stability guarantees, not tuning knobs. You can always make the recall metric move by loosening the bounds until the substrate is doing unconstrained Hebbian learning — that is a known-broken regime the field has already explored, and a result obtained that way is worthless even though the number improved. If you find yourself wanting to touch them, that is the signal to stop and write up why, not to proceed.

Consult `DESIGN.md` "Failure modes to expect" before diagnosing anything as a bug. Gates collapsed to 0, fast weights pinned at budget, and out-of-context recall at chance with hand-set gates are all documented expected behaviors with prescribed responses.

---

## 4. Cleanup rules

- Green invariant suite = cleanup. Touched invariant suite = design change → stop and ask the human.
- Style, logging, CLI, vectorization, and performance work on the tick are welcome, provided mutation stays tick-only (I-6) and the suite stays green.
- Never convert the runtime buffers (`fast`, `trace`, `consol`) to Parameters, never register them with any optimizer, never let any loss reach them (I-4).
- Do not add dependencies beyond torch and the standard library without reporting first.
- After ANY edit session, end by running the invariant suite and the `--fast` experiment, and report all three §2 numbers, even mid-task.

---

## 5. Reporting

When done (or blocked), report:
- The three §2 numbers across seeds, in a table.
- Every hyperparameter changed from defaults, with the §3 rung it corresponds to.
- Any file outside `scripts/` you modified, with justification per §1.
- Wall-clock and memory footprint of one meta-training run (needed for scale-up planning).

Blocked means: you have exhausted §3 rungs 1–5, or the task appears to require violating an invariant, or baseline numbers drift and you cannot isolate why. Report the state honestly rather than forcing the metric. A clean negative result ("meta-training at this scale does not lift out-of-context recall within these constraints") is an acceptable and reportable outcome — a contaminated positive is not.
