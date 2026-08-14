# Glial Substrate Prototype

Runtime-plastic sidecar substrate for transformer FFN weights: eligibility traces from local activity, glial-style group gating, per-group strength budgets, live consolidation. Backprop demoted to the outer loop. See `DESIGN.md` for the contract — **read it before editing**.

## Layout

```
substrate/
  layer.py     SubstrateLinear + GateNet + SubstrateConfig (the core)
  model.py     TinyTransformer (substrate in FFN) + SubstrateController
  tasks.py     AssociativeRecall + ContinualLM synthetic benchmarks
scripts/
  test_invariants.py   the contract, mechanically checked — run after ANY edit
  run_experiment.py    pretrain -> gradient-free runtime exposure -> probes
monitor/
  monitor.html         read-only live dashboard (self-contained, no deps)
  MONITOR_RULES.md     observer boundary rules for the agent — read before UI work
DESIGN.md      invariants, tunable surface, open work, expected failure modes
AGENT_ORDERS.md  post-build standing orders for the coding agent
```

## Watching a run

```bash
# terminal 1 — from the repo root:
python scripts/run_experiment.py
# terminal 2 — from the repo root:
python -m http.server 8137
# then open http://localhost:8137/monitor/monitor.html
# (or http://<jennybrain-ip>:8137/monitor/monitor.html from another machine)
```

The dashboard polls `metrics.jsonl` every 2s: loss curves for both models, gate
activity, fast-weight growth, consolidation, the neuromodulator, probe results
vs chance, and health badges mapped to DESIGN.md's documented failure modes.
It is a pure observer — deleting `monitor/` changes nothing about a run.

## Quickstart

```bash
pip install torch
python scripts/test_invariants.py        # must print "all invariants hold"
python scripts/run_experiment.py --fast  # ~2 min CPU smoke run
python scripts/run_experiment.py         # full toy run (GPU recommended)
```

## What the experiment measures

Both models (substrate vs parameter-matched frozen baseline, identical init):

1. **Pretrain** (backprop): learn the key→value recall *format* on half the key pool.
2. **Runtime exposure** (zero gradients): stream sequences using held-out keys with a novel fixed mapping. The substrate model accumulates traces and ticks; the baseline just runs forward.
3. **Probes**: (a) in-context recall on both pools, (b) **out-of-context recall** — query a held-out pair *not shown* in the sequence. Only persistent runtime memory can answer (b). Forgetting = pretrain-pool accuracy drop.

Chance = 1/16. The headline metric is the substrate-minus-baseline gap on (b) after exposure, and the forgetting cost paid for it.

## Current status

- All 9 invariants pass; pipeline runs end-to-end; substrate state verifiably
  moves under gradient-free exposure without destabilizing (smoke run:
  +6 pts in-context held-out gain vs +1 baseline; out-of-context at chance).
- Gates are hand-initialized. **Out-of-context recall is expected to stay near
  chance until the meta-training loop is built** (DESIGN.md, open work #1).
  That loop is the experiment.

## Hardware notes

The toy runs anywhere. The tick is O(params) with dense ops — fine at this scale, and structured so the forward path never leaves dense matmul (the two-timescale rule, invariant I-6). Scale-up guidance is in DESIGN.md open work #4: don't retrofit a real model until the toy shows a meta-trained effect.
