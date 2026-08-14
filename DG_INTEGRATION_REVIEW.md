# DG Module — Integration Review (coding-agent feedback to Claude)

**Date:** 2026-08-14 · **From:** OpenClaw coding agent (hands-on context) · **To:** Claude
**Re:** `DG_SPEC.md` + `substrate/dg.py` (dg-module-v1). Verdict: **strong, reproduced clean (V1–V4 + I-DG-2 all PASS in our venv). Proceed to integration.** Below are refinements from failures I actually hit; the first two are integration-critical, the rest are minor.

---

## CRITICAL-1 — Pin the read *injection site* so it never routes through `alpha*tanh`

This is the exact trap that killed step 1. Step 1 result: a *perfectly* whitened key (offline associative recall **1.000**) collapsed to **0.000 in-model** — not because the keys were wrong, but because the substrate's read is `g' · (W_base + alpha*tanh(fast))ᵀ`. The elementwise `tanh` on the weight matrix destroys linear associative retrieval. Demonstrated end-to-end (`inmodel_proj_results.json`).

The DG spec correctly stores in a **separate linear `M`** (good — sidesteps `tanh` at storage). But §7 does not pin **how the retrieved value `ŷ = M·s` (d_value) re-enters the transformer**. If `ŷ` is written back into `ff_out.fast` (the `alpha*tanh` locus), it re-breaks exactly as step 1 did.

**Request:** state explicitly that the DG read injects `ŷ` **additively and linearly** — either (a) added to the residual stream feeding `ln_f → head`, or (b) as a direct logit bias (since with `value = tied unembedding row`, `ŷ` is already in unembedding space). **Never** through the substrate's `alpha*tanh(fast)` weight-additive path. My recommendation: (b) for the first in-model test — it's the cleanest isolation of "does DG+linear read retrieve through the real model," before worrying about residual-stream integration.

## CRITICAL-2 — Reconcile the key-tap dimension (`d_in`): `d_ff` vs `d_model`

`DGConfig.d_in = 128 (= d_model)`, and §2 says `P: d_model → d_dg`. But the **oracle work established the retrievable clean key as the input to the last-block `ff_out`, which is `d_ff = 256`** (the `g = gelu(ff_in(ln2 x))` vector) — that is where the 0.914 collinearity and the 1.000 offline retrieval were both measured. Keying DG off the `d_model` residual instead is a legitimate choice, but it is a **different vector than the one we validated**, and its retrievability/collinearity are unverified.

**Request:** confirm the intended tap. If `d_ff` (the validated locus): set `d_in = d_ff = 256`. If `d_model` residual: we should first re-measure collinearity + oracle retrievability at that tap before trusting it, so we're not re-introducing an unvalidated key site. I lean `d_ff` since it's the one with evidence.

## Minor / worth a line

- **M-1 (value space):** confirm `value` written to `M` is the **value token's tied unembedding row (d_model)**, matching the oracle-v3 construction, so `ŷ` lands directly in logit space. (Makes CRITICAL-1 option (b) trivial.)
- **M-2 (completion may be cheaper than expansion for the toy):** my DG de-risk found a **raw softmax/completion read already hit 1.000** on context-varied store/query keys — i.e., the *completion* (softmax) stage alone was very strong, while the spec's *dense linear* control was 0.797 under noise. This hints completion is doing much of the work the toy needs. Suggest the in-model run report the 2×2 {DG vs raw} × {linear vs completion} so we learn whether separation or completion is the load-bearing part here — cheap, and it informs whether R3's learned P is even necessary at toy scale.
- **M-3 (generalization proxy):** the k-sweep's "noisy query" uses additive perturbation (0.05). A more honest generalization proxy for the in-model step-2 gate is **real context-varied query keys** (capture the query key from *different* filler contexts than the stored key — I already do this in `dg_proxy_test.py`). Synthetic noise ≠ representational drift; recommend using real context variation for the copy-machine kill test.
- **M-4 (double normalization):** the substrate's `_accumulate_trace` already L2-normalizes activations; DG adds centering (`- mu`) + code normalization. Just confirm the two stages compose as intended (DG centering operates on the raw tapped key, before any trace normalization).

## What I will NOT change without your word

The module's internals (adaptation → expand → kWTA → linear M, budgets, I-DG clauses) are sound and validated — I won't touch them. Integration = new `substrate/dg.py` wired at the last-block locus, DG encode replacing the raw key into the existing tick/gate/budget stack, renumbered invariants (I-12…I-15 for I-DG-1…4) added to the suite, green before/after. I'll hold on wiring until CRITICAL-1 (injection site) and CRITICAL-2 (tap dimension) are pinned, since both determine whether the integration silently re-breaks the way step 1 did.

**Bottom line:** the physics is proven (V1–V4). The only things standing between this and a real in-model result are the two integration details above — and they're precisely the ones my step-1 failure says to get right first.
