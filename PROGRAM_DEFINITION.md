# Glial Substrate — Program Definition (steering doc)

**Date:** 2026-08-14 · **Owner:** Rebecca · **Maintained by:** OpenClaw coding agent
**Purpose:** the map we steer from. What we HAVE (verified), what we WANT (the claim), what we NEED (gaps → borrow-or-build → test). Updated as tests land. This is the doc to work through with Claude; the handoffs (`ORACLE_*`, `RESEARCH_DIRECTION_*`) are the evidence behind it.

---

## THE CLAIM (what we want) — one sentence

**A self-describing memory as permanent architecture: a system that (1) learns continuously without catastrophic forgetting, (2) retrieves relevant stored information by content and generalizes to novel queries, and (3) knows what it knows (calibrated confidence) — all at runtime, without backprop.**

The novelty is not any single item — those exist separately. It is the **combination in one permanent, runtime-writable, self-monitoring system.** Recall alone is not the prize; the field has recall.

**Success = one integrated test where four signals move together** (the "four-in-one"):
1. **Acquisition** — learns new associations from a runtime stream.
2. **Retention** — keeps old ones (no catastrophic forgetting).
3. **Generalization** — retrieves correctly on *held-out* queries it was never fit to (real codec, not a copy machine).
4. **Calibration** — its confidence signal correlates with whether it is actually right (knows what it knows).

If these move together on the toy, the combination claim is demonstrated. If not, we learn exactly which link breaks.

---

## WHAT WE HAVE (verified this session)

**Infrastructure & guarantees (solid):**
- Toy: TinyTransformer with substrate sidecar in the FFN. **13 invariants green** (incl. I-11 functional-forward parity). Stability guarantees hold — bounded fast weights, per-group budget, gradient firewall; never diverges.
- Meta-training loop built (`meta_train.py`, v2): trains gates + `log_plastic_lr` through a first-class functional path; gradient-firewall self-guard verified (GateNet always gets nonzero grad).
- Observability: threaded monitor, dose-response card, mandatory pre-run UI verification (§0a).
- Clean provenance chain, every retraction documented.

**Diagnostic findings (what actually works / fails):**
- **Probe harness is valid** (KV oracle = 1.000). Every 0.000 is real.
- **Writing works; reading is the bottleneck.** Perfect, clean writes deliver signal (+3 logits to the target) but recall stays 0.000 in-model.
- **Root cause found:** the model's query-position "key" vectors for distinct facts are **~91% collinear** (mean |cos| 0.914). The additive outer-product read blends all stored values → non-selective.
- **It is fixable, not fundamental:** the associative read retrieves at **1.000** with orthonormal keys, and **whitening the existing keys restores 1.000** (offline). A key-decorrelation/projection step is provably sufficient in isolation.

**What we do NOT have yet:**
- Generalization (retrieval on novel/held-out keys) — untested.
- Continuous learning without forgetting — meta-runs showed **catastrophic forgetting (0.77–0.88)**.
- Any metacognition / calibration read-out — untouched (signals exist, unused).
- Any head-to-head vs standard methods — never built.
- In-model confirmation that the key-projection fix survives the full transformer path.

---

## WHAT WE NEED — gaps → BORROW-FIRST → build → test

Guiding rule (AGENTS.md "Existing Solutions Preflight"): **spend the novelty budget on the combination, not on re-deriving solved parts.** For each gap, borrow a proven pathway before inventing.

### Gap 1 — Selective read / retrieval (the current blocker)
- **Borrow first:** this is solved territory. **Modern Hopfield networks** (Ramsauer et al. 2020) are associative memories with huge capacity whose retrieval *is mathematically attention* — one-step, content-addressed. **Fast-weight programmers** (Schmidhuber; Schlag et al.) and **kNN-LM / Memorizing Transformers** all implement exactly "store key→value, retrieve by relevance." Associative-memory theory already says: **keys must be near-orthogonal** for capacity — which is precisely our 0.914-collinearity failure.
- **Design choice for Claude:** either (a) add a **learned key projection** (decorrelating map, meta-trained) in front of the existing additive read, or (b) **replace the additive-FFN read with a proven associative read** (Hopfield/fast-weight formulation) as the substrate's memory primitive. (a) preserves the current locus; (b) borrows a known-good pathway wholesale.
- **Tests:** in-model fixed-whitening confirm → then generalization (retrieve held-out keys).

### Gap 2 — Continuous learning without forgetting
- **Borrow first:** **EWC** (our `consol` is already EWC-like — lean into the real formulation), **experience replay / rehearsal**, **orthogonal-gradient / OGD** methods, **progressive/lateral** approaches. Continual-learning is a mature subfield; we should adopt, not reinvent.
- **Build:** make consolidation protect *function* (output behavior), not just fast-weight norm. Possibly interleave rehearsal in the deployment stream (the meta-run's train/deploy horizon mismatch pointed here).
- **Test:** retention of pretrain-pool while acquiring held-out pool (the forgetting metric we already log).

### Gap 3 — Knows-what-it-knows (metacognition / calibration)
- **Borrow first:** **temperature scaling**, **conformal prediction**, **selective prediction / deep ensembles** for calibrated confidence and abstention.
- **Build (the novel bit):** expose the substrate's *existing* signals as confidence — **surprise/neuromodulator** ("is this new?") and **consolidation** ("how established?") — and calibrate them. This is where our design is genuinely differentiated: the confidence read-out comes *free* from the plasticity machinery.
- **Test:** does confidence correlate with correctness (reliability diagram / AUROC of "will it be right")?

### Gap 4 — Proof it beats the shelf
- **Borrow first:** use off-the-shelf implementations as comparator arms — **in-context vanilla**, **RAG/kNN retrieval**, **runtime LoRA** (the capability ceiling, uses backprop).
- **Build:** matched harness — same base model, same data, same **compute budget**, report capability **and** cost (the frontier, not a single number).
- **Test:** does the substrate reach comparable capability at its constraint profile (permanent, runtime-writable, no backprop, self-monitoring) that none of the others offer together?

---

## SEQUENCING (how we actually get there — iterative, test-gated)

Each step is cheap on the toy and gated by a measurable result before the next.

1. **Confirm the read fix in-model** (Gap 1a): apply calibrated whitening at read inside the real forward; expect recall > chance. *Gate: does the offline 1.000 survive the transformer?*
2. **Generalization test** (Gap 1): train/fit the projection on some keys, test on held-out keys. *Gate: copy-machine vs real codec.* — this is the make-or-break for the whole idea.
3. **Borrow-vs-build decision** on the read primitive (projection vs Hopfield/fast-weight), based on 1–2.
4. **Forgetting** (Gap 2): add rehearsal/functional consolidation; *gate: retention while acquiring.*
5. **Metacognition** (Gap 3): expose + calibrate surprise/consol; *gate: confidence↔correctness.*
6. **Four-in-one integrated test** — the claim. All four signals together on a continual stream.
7. **Comparator suite** (Gap 4) — only once 6 shows an effect; then decide scale-up (respect DESIGN.md #4 gate).

---

## HONEST RISKS (so we kill it early if we must)

- **Generalization is the crux.** If step 2 shows the codec only works on fitted keys, the recall half is a copy machine and the program narrows to metacognition-only — decide then.
- **Recall alone is not novel.** If we can't get to the *combination* (esp. no-forgetting + calibration together), a working retrieval demo duplicates existing methods.
- **Borrowing may reshape the thesis.** If the proven read primitive is Hopfield/attention-like, the "glia/no-backprop" framing must still add something (permanence + self-monitoring + continual write) or the novelty erodes. Keep asking: *what does this do that RAG+LoRA together don't?*

---

## OPEN DECISIONS FOR CLAUDE

- **P1** — Accept the four-in-one as the definition of success? Adjust the four?
- **P2** — Gap 1: learned projection in front of the additive read, or adopt a Hopfield/fast-weight read primitive? (borrow-vs-build)
- **P3** — Which borrowed pathways to pull in for Gaps 2 and 3, and are any incompatible with the invariants (I-4 firewall, I-5 locality, I-6 tick-only)?
- **P4** — Confirm sequencing; specifically that generalization (step 2) is the early make-or-break gate.
- **P5** — What single result would make you say "this combination is real," and what single result would make you say "kill it"?

---

# RULINGS ABSORBED � PROGRAM REVIEW P1�P5 (2026-08-14, Claude; Rebecca overrides)

**P1 � four-in-one ACCEPTED + two amendments.**
- Pre-registered bars (revisable by Rebecca BEFORE the integrated test, never after): (1) acquisition oo-ctx >= 0.50; (2) retention forgetting < 0.10; (3) generalization held-out/fitted retrieval ratio >= 0.80; (4) calibration AUROC >= 0.75 and ECE <= 0.10 after one offline scaling pass. All simultaneous, 3 seeds, one continual stream.
- Fifth clause = CONSTRAINT PROFILE, stated with every result: no runtime gradients; bounded state (I-1/I-2); permanent architecture (no external store at inference); self-gated (no human/task write boundaries). This is what RAG+LoRA+temp-scaling cannot jointly offer.

**P2 � projection first, built so it can become Hopfield.** Learned map P before write AND read (same space both sides); prefer EXPAND + SPARSIFY (dentate-gyrus pattern separation), not mere rotation; meta-trained, FROZEN at runtime (extend I-4 firewall to P). Copy-machine guard: P is one fixed function of any input, never fit per-key. Escalation (pre-registered): if in-model projection fails step 1 or codec ratio fails step 2, add softmax score-sharpening (=> full Hopfield read) as the second and FINAL Gap-1 attempt; if both fail generalization, P5 kill applies.

**P3 � borrow list filtered by invariants (offline/outer-loop borrowing OK; runtime forward-only).**
- Forgetting: ADOPT rehearsal/replay (interleave into deployment; also fixes train/deploy horizon mismatch). ADOPT EWC-idea not impl: consol becomes FUNCTIONAL importance from forward-only signals (logit-margin with group fast-contrib present vs suppressed, 2 forward passes at tick). REJECT OGD (runtime grads). DEFER progressive nets (violates permanence).
- Calibration: ADOPT temperature scaling (meets ECE bar) + conformal prediction (forward-only thresholds => free abstention). REJECT deep ensembles. BUILD (crown jewel): confidence = learned+calibrated readout of match-strength-in-P-space + consolidation depth + surprise/neuromod history. PROMOTE calibration-from-physiology to co-headline with the combination.

**P4 � sequencing confirmed + two free insertions.** Generalization (step 2) is the early make-or-break gate. Insertion 1: log RETENTION continuously from step 1. Insertion 2: log raw calibration signals (match strength, consol, surprise) UNCALIBRATED from step 1, so step 5 calibrates against accumulated history.

**P5 � the two sentences.**
- REAL: one continual-stream run, 3 seeds, all four bars met simultaneously under the constraint profile, WITH the ablation arm (random/hand-fixed gates) failing >= 2 of 4 (proves learned control, not mere plasticity).
- KILL: generalization gate fails (held-out retrieval ~chance while fitted succeeds) after BOTH read primitives (projection, then projection+sharpening) tried => copy-machine, terminal for memory claim; program narrows to calibration-only or closes. Secondary kill: any pair of the four trades ~1:1 after boxed attempts => combination dies as a measured incompatibility (itself a finding).

Stopping machinery (RULINGS_META_TRAIN �4 / RULINGS_3 �5) remains in force; nothing here extends the ladder.
