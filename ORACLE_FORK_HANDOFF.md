# Oracle Fork — Result Handoff (RULINGS_2 R6)

**Date:** 2026-08-14
**From:** OpenClaw coding agent (Claude Opus, under AGENT_ORDERS.md), with Rebecca
**To:** the designing assistant (Claude)
**Prereqs:** RULINGS_2_RESEARCH_DIRECTION.md (R6 = "run the fork before any new architecture"), RESEARCH_DIRECTION_HANDOFF.md
**Code:** `scripts/oracle_probe.py`, results in `oracle_results.json`, logged to `metrics.jsonl`

---

## 0. TL;DR

The fork ran. **The harness is valid and the failure localizes to RETRIEVAL, not write-selection.**

- **Metric-sanity control: PASS = 1.000.** A model-independent exact-match KV table answers out-of-context recall perfectly. The probe harness CAN leave chance; every 0.000 to date is a real measurement. This question is permanently closed.
- **Oracle-trace arm: 0.000 out-of-context, at every alpha and under two independent faithful binding specifications.** Perfect, smear-free writes are unreadable. → Gap = **RETRIEVAL (D-squash / D-faint)**, per your fork.

But there are two confounds I did NOT paper over (§3) that shape what "retrieval gap" is allowed to mean, and one of them points at a *structural* cause (the I-2 budget), not just I-1's alpha.

---

## 1. What was run

Full scale: d_model=128, 4 layers, d_ff=256, 3000 pretrain steps (model learns the format cold — pretrain loss ~1e-4). Chance = 0.0625.

- **Arm 1 — metric sanity (`KVOracleModel`).** Model-independent; emits the true value token at the probe position from an exact table. Validates that the probe format is answerable and the metric measures it.
- **Arm 2 — oracle-trace.** Bypasses learned gating and Hebbian accumulation; writes into `fast` the ideal binding outer product for each held-out pair, **binding pair only, no smear**, gate fully open, then the standard per-group budget rescale. Then the standard out-of-context probe. Swept alpha ∈ {0.1, 0.2, 0.4} plus a linear-read variant.

**Two binding specifications tried (I distrusted my own first oracle):**
- **v1 binding:** pre at the in-context KEY position, post at the in-context VALUE position (cross-position). → 0.000.
- **v2 binding (corrected, faithful):** pre AND post at the trailing QUERY position during an in-context-correct forward (pair shown, so post genuinely emits the value). This is the substrate's own same-position Hebbian rule `outer(post, pre)` keyed on "querying k → emit v." → **still 0.000.**

Both specs, every alpha, linear-read: out-of-context recall = 0.000. KV sanity = 1.000.

---

## 2. Results

| arm | oo-ctx recall | note |
|---|---:|---|
| KV metric-sanity | **1.000** | harness valid |
| oracle, α=0.1 | 0.000 | in-ctx 0.325 |
| oracle, α=0.2 | 0.000 | in-ctx 0.388 |
| oracle, α=0.4 (4× the I-1 bound) | 0.000 | in-ctx 0.367 |
| oracle, linear-read (scale 0.15, α=0.4) | 0.000 | in-ctx 0.297 (see confound §3.1) |

**Notable secondary signal:** oracle in-context recall is only ~0.30–0.39 — *lower* than the trained model's native in-context (~0.5–0.6). The oracle write does not merely fail to help out-of-context; it slightly **degrades** in-context. An injected FFN-weight-space association, however constructed, is not behaving like retrievable KV memory here — it's mild noise on the forward pass. That is an architectural signal, not a tuning signal.

---

## 3. Confounds — read before trusting the verdict

**3.1 The linear-read arm is confounded and should be considered inconclusive.** I implemented "near-linear tanh" by shrinking `fast` (scale 0.15) — but that also shrinks the write magnitude, so it conflates *linearity* with *less signal* (worsens D-faint while testing D-squash). A clean linear-read test must hold contribution magnitude constant while pushing tanh into its linear region, which the budget rescale makes awkward. **Do not weight the linear-read 0.000.** The solid evidence is the alpha sweep (0.1→0.4, all 0.000).

**3.2 The per-group budget (I-2), not alpha (I-1), may be the structural cap.** The budget rescale caps each group's fast contribution at `budget_frac · ||W_base_group|| = 0.25 · ||W_base_group||`, *independent of alpha*. So raising alpha cannot increase the retrievable signal beyond a 25%-per-group nudge to `W_eff` — higher alpha just gets rescaled back down. The alpha sweep showing 0.000 across 0.1–0.4 is therefore partly *expected*: I-2 dominates I-1 here. **This means the honest evidence package is about `budget_frac` (I-2) at least as much as `alpha` (I-1)** — and both are stability guarantees needing your sign-off, not agent tuning. I did NOT touch either in any learning run; the sweep changed alpha in the oracle arm only, as authorized.

**3.3 Deeper hypothesis this raises.** The convergent 0.000 across two faithful writes, plus the in-context *degradation*, suggests the blocker may be the **architectural choice itself**: additive, tanh-bounded, budget-capped fast weights in the *FFN weight space*, read by the same dense matmul, may not implement content-addressable retrieval at all — regardless of how perfectly the write is specified. If so, this is bigger than an I-1/I-2 number: it questions whether the substrate's *read* (a bounded additive weight perturbation) can ever surface a stored association at the head. That would push toward a **separable read path** (RULINGS_2 R2b query-trace alignment was heading here) or a different memory locus — not just a bounds revision.

---

## 4. Decision requests

- **F1 — Verdict acceptance.** Do you accept "retrieval-limited, not write-limited" given two faithful oracle specs at 0.000 and KV-sanity at 1.000? Or do you see a third oracle binding worth trying before concluding (e.g. writing into `ff_out` value-space specifically, or binding to token-embedding directions rather than FFN activations)?
- **F2 — I-2 vs I-1.** Given §3.2, should the evidence package target `budget_frac` (I-2) rather than / in addition to `alpha` (I-1)? What controlled oracle sweep over `budget_frac` would you accept as evidence for a considered I-2 revision (with Rebecca's sign-off)?
- **F3 — Linear-read redo.** Want a corrected linear-read arm (constant contribution magnitude, tanh linearized) before the retrieval verdict is final, or is the alpha/budget analysis sufficient?
- **F4 — Architecture implication (§3.3).** Does the in-context *degradation* + convergent 0.000 move you toward a separable read path / different memory locus now, or do you want the budget-sweep evidence first?
- **F5 — Next build.** RULINGS_2 R2 assumed the fork would show write-selection (→ write-salience). The fork shows retrieval. Does R2 change? My read: build the **separable/aligned read** (R2b) BEFORE write-salience (R2a), since salience only helps if writes are readable — and right now they are not.

---

## 5. State / reproduction

```
.\.venv\Scripts\python.exe scripts/test_invariants.py     # all invariants hold (13, incl I-11)
.\.venv\Scripts\python.exe scripts/oracle_probe.py        # full fork (~1 min GPU)
.\.venv\Scripts\python.exe scripts/oracle_probe.py --fast # smoke (under-trained; do not trust verdict)
```

- No invariant changes. `alpha`/`budget_frac` untouched in every learning run; oracle-arm alpha sweep is diagnostic only, restored after each measurement.
- DESIGN.md failure-modes updated (R7): volume-knob-vs-router + D-smear/D-squash/D-faint recorded verbatim.
- Provenance chain intact: v1 negative → v2 corrected → oracle fork (localized). This sequence is the publishable spine.

**Caveat carried forward:** smoke-scale (`--fast`) oracle showed a spurious 0.127 because the base model was under-trained; the trustworthy verdict is the full-scale 0.000. Always run the fork at full pretrain.
