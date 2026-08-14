# Oracle V2 Diagnostic — the Verdict Flips to FIXABLE (RULINGS_3 V2)

**Date:** 2026-08-14
**From:** OpenClaw coding agent (Claude Opus), with Rebecca
**To:** the designing assistant (Claude)
**Code:** `scripts/oracle_diag.py`; results `oracle_diag_results.json`; logged to `metrics.jsonl`
**Supersedes the lean of:** `ORACLE_V3_HANDOFF.md` (which leaned F4-fail). This overturns that lean with direct evidence.

---

## 0. TL;DR

You called the sub-fork (V2): before declaring the additive locus dead, test whether the read fails because the *keys overlap* (fixable) or the *locus is fundamentally wrong* (not). We ran it. **It is key-orthogonality. Fixable.**

- **[A] key overlap:** distinct held-out clean keys are **~91% collinear** (mean |cos| 0.914, max 0.947).
- **[B] offline read, real keys:** 0.125 (≈chance) — confirms non-separable.
- **[C] offline read, orthonormal keys:** **1.000** — the outer-product read mechanism is perfect when keys are separable.
- **[C'] offline read, *whitened* real keys:** **1.000** — merely decorrelating the *existing* keys restores perfect recall.

**Conclusion:** the fast-weight associative memory is sound. The blocker is that the pretrained model's query-position key representations are nearly collinear (dominated by the shared "being-queried" / format direction). A **learned key projection** (decorrelating map) before write and read is provably sufficient in isolation — and it is exactly what the meta-loop can learn. Do NOT abandon the additive locus; do NOT relax I-1/I-2.

---

## 1. What was run (pure measurement, no learning, no bound/invariant changes)

Last-block `ff_out`, clean keys from pair-absent forwards (the V3-correct construction). Then the outer-product associative read isolated from the transformer:
`M = Σ_k outer(emb[value_k], key_k)`; for query k′, `argmax_v emb_v · (M @ key_k′)`.

- **A** — pairwise cosine of the clean keys (separability).
- **B** — offline read with the real keys (real-key separability through pure associative math).
- **C** — offline read with an orthonormal key set from QR of the real-key span (mechanism ceiling).
- **C'** — offline read with whitened real keys (`G^{-1/2}` decorrelation) (does unmixing the *existing* keys suffice?).

Full scale: d_model 128, 4 layers, 3000 pretrain steps, pretrain loss ~1e-4, seed 0.

---

## 2. Results

| probe | recall | reading |
|---|---:|---|
| [A] key cosine (mean / max) | 0.914 / 0.947 | keys nearly parallel — not separable |
| [B] offline read, real keys | 0.125 | ≈ chance (0.062); real keys can't be told apart |
| [C] offline read, orthonormal keys | **1.000** | read mechanism works perfectly when keys separable |
| [C'] offline read, whitened real keys | **1.000** | decorrelating existing keys is sufficient |

In-model Oracle v3 recall was 0.000 with logit_shift rising to +3 — consistent: the write raises the target logit but also all competitors (blended by ~0.91-collinear keys), so argmax never flips.

---

## 3. Interpretation

The query-slot representation for "QUERY k" is dominated by a shared component (the format/positional "I am being asked" direction); only a small residual encodes *which* key. So `key_k · key_k′ ≈ 0.91` for all pairs, and the outer-product read returns a near-uniform blend of stored values. Orthogonalizing (C) or decorrelating (C') the keys exposes the discriminative residual and retrieval becomes perfect. This is the classic linear-associative-memory requirement (near-orthogonal keys) not being met by raw pretrained representations — a well-understood, fixable condition, not a failure of the storage locus.

---

## 4. Decision requests

- **W1 — Accept the flip?** Do you accept "key non-orthogonality, F4-fixable" over "additive locus dead," given [C]=[C']=1.000 vs [B]=0.125 and mean|cos|=0.914?
- **W2 — Next build = learned key projection.** Proposed: a small trainable linear projection `P` mapping the `ff_out` input (query representation) into a decorrelated key space, used for BOTH the write (key stored = `P·key`) and the read. Trained by the existing meta-loop (backprop reaches it through the functional path, like GateNet). It respects I-4 (meta-only gradients), I-5 (local: it's a function of the layer's own input), I-6 (used at tick/read, not mutating buffers in forward). Do you want `P` per-substrate-layer or only at the read locus (last `ff_out`)? Linear, or a tiny MLP? Whitening-initialized (`G^{-1/2}` from a calibration pass) or random?
- **W3 — In-model confirmation.** The 1.000 is offline (isolates the associative math; no tanh/budget/residual/ln/head competition). Before building the full learned projection, do you want a cheap **in-model oracle-with-fixed-whitening** run (apply the calibrated `G^{-1/2}` at read inside the real forward) to confirm retrieval survives the transformer path at bounded dose? My recommendation: yes — it's the minimal step that proves the fix works end-to-end before committing the meta-loop to learning `P`.
- **W4 — Invariant.** If a key projection enters `substrate/`, propose its invariant (e.g. I-12: projection is a function of local input only, gradients reach it only in meta mode, it does not alter the pretrain-mode forward). Needs your + Rebecca's sign-off as a design change.

---

## 5. State / provenance

- No invariant changes; contract green (13 checks incl. I-11). Diagnostic is measurement + offline linear algebra only; no learning run; no bound touched.
- DESIGN.md updated: key-non-orthogonality finding + fix recorded.
- Monitor unchanged (diagnostic logs a `diag` row; dose-response card already added for observability).
- Provenance spine: v1 negative → v2 corrected → oracle v1/v2 (contaminated, retracted) → oracle v3 (clean: read non-selective) → **V2 diagnostic (clean: cause = key collinearity, fix = key projection)**. The story is now a diagnosis with a concrete, evidence-backed remedy.

**Bottom line:** the idea's memory works. It needs a key-sharpening (decorrelation) stage, which the meta-loop can learn. That is the next build, pending your W1–W4.
