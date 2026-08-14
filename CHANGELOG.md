# Changelog

Provenance for the glial-substrate research program. Newest first.
Semantic-ish: MAJOR.MINOR.PATCH-stage. This is a research prototype; "alpha"
means the core science claim (the four-in-one) is not yet demonstrated.

---

## v0.3.0-alpha1 — "Read Organ Online" (2026-08-14)

First version where out-of-context retrieval works in-model and is audited real.
Session with Rebecca McClintic + designing assistant (Claude); coding by OpenClaw
agent (Claude Opus).

### Added
- `scripts/meta_train.py` — meta-training loop for plasticity gates (DESIGN open
  work #1), v2 with surprise neuromod + retention term. v1 preserved as
  `scripts/meta_train_v1_negative.py`.
- `substrate/model.py` — first-class functional forward path (`fast_map`/`act_out`)
  for meta-mode; bit-identical to runtime when unused (invariant I-11).
- `substrate/dg.py` — dentate-gyrus module (adapt→expand→kWTA→linear assoc mem)
  from the designing assistant; `scripts/dg_validate.py` acceptance gates V1–V4.
- Oracle diagnostics: `oracle_probe_v3.py` (clean positive control), `oracle_diag.py`
  (key-orthogonality), `inmodel_proj_test.py`, `dg_inmodel.py` (2×2 read study),
  `dg_controls.py` (contamination audit), `dg_sweep.py` / `dg_autotune.py`
  (viability + auto-tuning), `load_compare.py` / `hybrid_compare.py` (load×noise).
- Invariants I-11..I-16 added (functional-forward parity, DG firewall, DG shared
  space, DG locality, DG per-column budget, DG retrieval-linearity dead-path ban).
  Full suite: 18 checks, all green.
- Monitor: threaded `monitor/serve.py`; dose-response, DG-sweep, and load-test
  cards; fetch-path fix (`../metrics.jsonl`).
- Observability rule codified in `AGENT_ORDERS.md §0a`.

### Findings (the scientific spine)
- Meta-loop alone: out-of-context recall stayed at chance + catastrophic
  forgetting (0.77–0.88). Honest negative.
- Oracle fork: writing works; **reading was the bottleneck**. Probe harness valid
  (KV oracle 1.000). Perfect clean writes → 0.000 in-model.
- Root cause: query-position keys ~91% collinear (transformer anisotropy) →
  additive outer-product read blends all values.
- **Dead-path law (I-16):** associative retrieval routed through the substrate's
  `alpha*tanh(fast)` weight-additive read is destroyed by the elementwise tanh.
  Whitened key: 1.000 offline → 0.000 in-model. Retrieval must be linear
  end-to-end.
- **Breakthrough:** store in a separate associative memory + inject retrieved
  value as a **linear logit bias** → out-of-context recall 0.000 → ~1.0 in-model.
  Contamination controls (empty mem, permuted binding, shuffled query all →chance)
  confirm real retrieval, not leakage.
- **Synthesis result:** across load×noise, the winning read is `dg_dense`
  (random expansion + softmax completion, NO hard sparsify): avg recall 0.997,
  beating raw softmax (0.914) and hard-DG (0.744). The expansion is the keeper;
  DG's biological sparsity was unnecessary/harmful for retrieval robustness.
- Auto-tuner: DG recall tunable to a stable plateau (mean 1.000, std 0.000);
  no active controller needed at toy scale.

### Honest scope (what is NOT yet established)
- Only single-token associative recall at toy scale (128-d).
- The working read is a NON-NOVEL mechanism (random features + attention /
  Hopfield-like). The win came partly from deleting the novel DG part.
- Writes here are oracle-style; the substrate's own gated Hebbian tick +
  meta-loop + **catastrophic forgetting** remain unsolved.
- Four-in-one claim (continuous learning + no forgetting + generalization +
  calibration): ~1 of 4 pillars banked (retrieval); the differentiated pillars
  (no-backprop continual learning, metacognition) are untouched.

### Contract
- `scripts/test_invariants.py` → `all invariants hold` (18 checks).
- Stability bounds (alpha, budget_frac) never modified in any learning run.

---

## v0.2 — received prototype (pre-session)
- TinyTransformer + substrate FFN, 9→13 invariants, meta machinery
  (`functional_tick`) present but no outer loop. Out-of-context recall at chance
  by design (hand-set gates). Baseline documented in `results.json`.

## v0.1.0 — original substrate (`substrate/__init__` legacy tag)
