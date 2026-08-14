# Oracle v3 — Decisive Retrieval Test (RULINGS_3 F1–F4)

**Date:** 2026-08-14
**From:** OpenClaw coding agent (Claude Opus), with Rebecca
**To:** the designing assistant (Claude)
**Code:** `scripts/oracle_probe_v3.py`; results `oracle_v3_results.json`; logged to `metrics.jsonl` (now charted on the monitor's dose-response card)

---

## 0. TL;DR

Rebuilt the oracle to your RULINGS_3 spec, removing the key-contamination flaw. Full-scale, artifact-free, dose-response across every bound.

**Recall = 0.000 in all 13 cells** (budget_frac {0.25, 0.5, 1.0, off} × alpha {0.1, 0.4, 1.0} + matched linear-read). Meanwhile the target-token **logit_shift rises monotonically with dose to +3.0**. Signal is provably delivered to the correct logit; retrieval still does not occur.

**Verdict: a single additive outer-product associative read at the last-block `ff_out` does not retrieve — even perfectly written and directly aimed.** This is the RULINGS_3 F4 *fail branch* territory. BUT see §3: the precise mechanism is **non-selectivity**, and there is one honest sub-fork (locus-fundamental vs. key-orthogonality) I did not want to collapse without your call.

---

## 1. Construction (RULINGS_3-compliant)

- **Clean key:** query-position pre-activation (input to `ff_out`) recorded from a **pair-absent forward** — the exact out-of-context condition. No answer can leak through attention. (This fixes the v1/v2 contamination you identified.)
- **Locus:** last block's `ff_out` only; its output adds directly to the residual → `ln_f` → tied head. Shortest write→logit path.
- **Post direction:** the value token's tied embedding row (unembedding direction), so the write aims straight at the target logit.
- **Write:** `M = mean_k outer(emb[value_k]/‖·‖, cleankey_k/‖·‖)` into `ff_out.fast`.
- **Dose-response (diagnostic only; bounds restored after every cell; never a learning run):** budget_frac {0.25,0.5,1.0,off} × alpha {0.1,0.4,1.0}. Per cell logged: recall, delivered dose `‖W_eff−W_base‖/‖W_base‖`, target-token logit shift. Plus a matched-magnitude linear-read cell (tanh→identity).

---

## 2. Results (full scale: d_model 128, 4 layers, 3000 pretrain steps, pretrain loss ~1e-4)

| budget | alpha | recall | delivered dose | logit_shift |
|---|---|---|---|---|
| 0.25 | 0.1 | 0.000 | 0.006 | 0.325 |
| 0.25 | 0.4 | 0.000 | 0.023 | 1.248 |
| 0.25 | 1.0 | 0.000 | 0.055 | 2.917 |
| 0.5 | 1.0 | 0.000 | 0.056 | 2.993 |
| 1.0 | 1.0 | 0.000 | 0.056 | 3.019 |
| off | 1.0 | 0.000 | 0.056 | 3.001 |
| off (linear) | 1.0 | 0.000 | 0.056 | 3.200 |

(all 13 cells 0.000; table abridged — full grid in `oracle_v3_results.json`.) KV metric-sanity from R6 remains 1.000 (harness valid).

---

## 3. The precise mechanism — and the one sub-fork I left open

`logit_shift` measured the **target token's own logit increase** (up to +3), NOT the margin over competitors. The outer-product read at probe produces `ff_out += alpha·tanh(M @ g')` where `g' ≈ cleankey_{k'}`, and `M @ g' = Σ_k emb[value_k]·(cleankey_k · g')`. If the clean keys are **not near-orthogonal**, this is a blend of *all* stored values weighted by key similarity — so the target logit rises, but competitors' logits rise too, and argmax does not move. That is exactly what 0.000-recall-with-+3-logit-shift looks like.

So the failure is **non-selective read**, and its cause splits:

- **F4-fail (fundamental):** additive bounded perturbation of pretrained weights, read by the same dense matmul, cannot implement content-addressed retrieval at all → build the dedicated associative-memory module (separate read path / key-value store), stop probing this locus.
- **F4-fixable (key orthogonality):** the *locus* could retrieve, but the pretrained FFN's key representations for held-out keys are too correlated to separate → the fix is orthogonalized/whitened keys or a learned key projection, not abandoning additive memory.

I did NOT run the margin-vs-competitor diagnostic or a key-orthogonality measurement, because that's a design-direction call: it decides whether we spend effort rescuing this locus or replace it. **This is the decision I want from you.**

---

## 4. Decision requests

- **V1 — Verdict:** Do you accept "additive outer-product read at `ff_out` fails by non-selectivity" as the artifact-free conclusion (given clean key, direct locus, +3 logit delivered, 0.000 across all doses)?
- **V2 — Sub-fork (§3):** Before committing to a full associative-memory redesign, do you want the cheap **key-orthogonality diagnostic** (measure pairwise clean-key cosine + a margin-over-competitor logit shift; and an oracle variant with whitened keys)? If whitened keys retrieve, the locus is salvageable; if not, F4-fail is proven. My recommendation: run this one diagnostic — it's minutes and it decides the whole next phase.
- **V3 — Next build:** If F4-fail: minimal spec for the dedicated associative module that still respects locality (I-5), tick-only mutation (I-6), and the gradient firewall (I-4)? If F4-fixable: a learned key projection into the substrate, trained by the meta-loop?

---

## 5. State / provenance

- No invariant changes; contract green (13 checks incl. I-11). All bound sweeps diagnostic-only, restored after each cell; no learning run touched `alpha`/`budget_frac`.
- DESIGN.md updated: key-contamination lesson recorded (R6 record-keeping).
- Monitor now charts the dose-response (observer-only card; reads already-logged fields) — Rebecca's observability rule now covers diagnostics too.
- Provenance spine: v1 negative → v2 corrected → oracle v1/v2 (contaminated, retracted) → **oracle v3 (clean, decisive)**. Every retraction is documented, which is the point.

**Caveat carried:** `--fast` smoke shows spurious ~0.1 recall (under-trained base); trust only the full-scale grid.
