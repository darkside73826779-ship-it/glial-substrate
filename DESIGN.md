# Glial Substrate — Design Contract

**Read this before editing anything.** This document exists so that code cleanup, refactoring, or extension does not silently destroy the design. The invariants below are the experiment. If an edit breaks one, the results stop meaning anything, even if the code still runs and the numbers still print.

`scripts/test_invariants.py` mechanically checks every invariant. **Any change to `substrate/` must end with that script passing.** If a refactor requires changing the invariant tests themselves, stop — that is a design change, not a cleanup, and it needs the human's sign-off.

## What this is

A prototype testing one claim: an LLM's weights, given a living sidecar substrate — eligibility traces written by local activity, group-level plasticity gates, per-group strength budgets, and consolidation shields — can acquire and retain information at runtime **without backpropagation**, demoting backprop to a meta-learning role (installing the plasticity rules, outer loop only).

The biological reference is not neurons-under-weights but **glia**: astrocyte-like units that each oversee a small cluster of synapses, gate their plasticity, enforce homeostasis, and never sit on the fast signal path.

Architecture summary: `SubstrateLinear` holds frozen base weights `W_base` plus three same-shaped runtime buffers — `trace` (Hebbian eligibility, written during forward), `fast` (runtime memory, written only at tick), `consol` (consolidation importance, written only at tick). Weights are grouped in blocks of `group_size` along fan-in; a shared `GateNet` maps per-group statistics plus one global neuromodulator scalar to a plasticity gate per group. The forward path is a single dense matmul over `W_base + alpha * tanh(fast)`.

## Invariants (the contract)

**I-1 — Bounded fast contribution.** The fast-weight contribution to any effective weight is `alpha * tanh(fast)` and can never exceed `alpha` in magnitude, no matter what the traces or gates do. This is the outermost stability guarantee. Never replace `tanh` with an unbounded function; never let `fast` enter the effective weight un-squashed.

**I-2 — Per-group strength budget.** After every tick, each group's fast-contribution L2 norm is rescaled to at most `budget_frac × ||W_base||` of that group. This is heterosynaptic normalization — the shared-resource budget that prevents Hebbian runaway. The rescale must happen *after* the update, inside the tick, every tick. Scale factors are clamped to ≤ 1 (the budget never amplifies).

**I-3 — Functional neighborhood, not positional.** Groups are contiguous blocks along the **fan-in dimension** of each output unit (dendritic-branch analogy). This is deliberate: adjacency in a raw weight matrix is meaningless (columns are permutable), so "nearby" must mean "cooperating on the same output." Do not regroup along fan-out, across rows, or by matrix position without understanding this changes the science. Any future lateral group-to-group communication must respect functional, not positional, structure.

**I-4 — Gradient firewall.** Four clauses:
  (a) Runtime state (`fast`, `trace`, `consol`) are registered buffers, never `nn.Parameter`. No optimizer may ever hold them.
  (b) At runtime, no gradients flow anywhere: trace accumulation and tick are `no_grad`.
  (c) `W_base` has `requires_grad=False` outside pretrain mode. Runtime never edits base weights.
  (d) Meta-training reaches `GateNet` and `log_plastic_lr` **only** through the functional mirror (`functional_forward` / `functional_tick`), which operates on passed-in tensors, never the buffers. Backprop installs the plasticity rules; it never performs the runtime learning. This is the entire thesis — break it and the experiment is measuring backprop again.

**I-5 — Locality of signals.** The trace is computed from the layer's own input and output activations only. The single permitted non-local signal is the broadcast neuromodulator scalar (one float, computed by the controller from prediction surprise). No targets, no labels, no loss values, no other layers' activations may ever feed a trace or a gate feature. If you want richer gating, add group-local statistics or widen the broadcast to a *small vector* of global scalars — never per-weight external signals.

**I-6 — Two timescales, one mutation site.** The forward path is a pure dense matmul plus (in runtime mode) trace accumulation. `fast` and `consol` mutate **only** inside `tick()`. Never move updates into the forward pass — the fast path stays dumb and dense (hardware reality), the slow path stays smart and sparse in time.

**I-7 — Inert during pretrain.** In pretrain mode the substrate does nothing: no traces, no ticks, `fast` stays zero. Pretraining must produce a model whose forward is bit-identical to a vanilla network so the comparison is clean.

**I-8 — Stability under sustained load.** Arbitrary input streams plus continuous ticking must never produce NaN/Inf or unbounded state. I-1 and I-2 guarantee this structurally; the test verifies it empirically. If tuning ever requires weakening I-1 or I-2 to get an effect, the effect is not real.

**I-9 — Fair baseline.** The baseline model shares the substrate model's forward-path parameter count and initialization seed. Substrate-only parameters (GateNet, log_plastic_lr) are excluded from parity because they never touch the forward path. Any benchmark change must preserve an apples-to-apples frame.

**I-10 — Consolidation shields, never amplifies.** Consolidation divides update magnitude (`delta / (1 + consol)`), grows only where gated updates agree in sign with accumulated fast weights, and decays slowly. It must never increase plasticity or be writable from outside the tick.

## Safe to change (tunable surface)

- All numeric values in `SubstrateConfig` (decays, rates, `alpha`, `budget_frac`, `group_size`, gate width). Tuning these is expected — stability tuning is the known time sink.
- `GateNet` architecture (depth/width/activation), provided it remains a function of the permitted features (I-5) and outputs (0,1).
- The group feature set — add group-local statistics freely.
- The neuromodulator formula in `SubstrateController` (keep it a small broadcast, per I-5).
- Tick scheduling (`tick_every`, adaptive schedules).
- Benchmarks, task generators, logging, CLI, performance optimizations of the tick (vectorization, chunking, fused ops) — provided mutation stays tick-only (I-6).
- Adding new benchmarks (ContinualLM in `tasks.py` is wired for a forgetting experiment the main script doesn't run yet).

## Known open work (intended next steps, not bugs)

1. **Meta-training loop.** `functional_tick` exists and gradients verifiably reach GateNet (I-4d test), but no outer-loop training script is included yet. This is the highest-value next build: episodes of (exposure → tick → probe) with probe loss backpropagated through the functional path to GateNet. Until then, gates run on hand-initialized rules and out-of-context recall staying near chance is expected, not a failure.
2. **ContinualLM benchmark script** using the two-grammar task.
3. **Lateral group communication** (the grid from the original sketch) — deliberately deferred until meta-trained gates establish a baseline effect. When added: functional neighbors only (I-3), slow-tick only (I-6).
4. **Scale-up** to a retrofit on a real frozen model (LoRA-style sidecar). Do not attempt until the toy shows a meta-trained effect.

## Failure modes to expect (so they aren't "fixed" wrongly)

- **Gates collapse to 0**: plasticity never turns on. Raise GateNet final bias, increase neuromod gain `k`. Not a bug in the gate code.
- **Gates saturate to 1 + fast pinned at budget**: runaway caught by I-2 doing its job. Lower `plastic_lr` / `consol_rate`; do not raise `budget_frac` first.
- **Substrate model slightly worse at pretrain**: should not happen (I-7 makes pretrain identical); if observed, suspect a mode-management bug.
- **Out-of-context recall at chance with hand-set gates**: expected. See open work #1.
- **Gate is a volume knob, not a router** (observed 2026-08-14, meta-train v1+v2): meta-training reliably moves the gate's write *magnitude* (v1 saturated ~0.97; v2 held ~0.44) but never demonstrates *selectivity* — out-of-context recall stayed at chance across every configuration and every §3-ladder setting. A scalar-per-group, surprise-gated gate can learn *how much* to write, not *what to write where*. No ladder rung converts a volume knob into a router; do not keep tuning for this. The addressing capability is what open work #3 (and write-salience, R2 of RULINGS_2) exists to add.
- **Address information is present but destroyed, not absent.** `outer(post, pre)` is the write rule of a linear associative memory and the dense matmul is its read rule — so binding address *is* encoded; three mechanisms destroy it before it can be read back:
  - **D-smear (temporal):** the trace superimposes outer products from every position of every sequence; the binding pair is buried under format tokens and hostile-phase content. Nothing selects the *binding moment* — the gate can only pass or block the whole smear. Prescribed response: per-position write salience (RULINGS_2 R2a), not a bounds change.
  - **D-squash (read nonlinearity):** superposition retrieval needs near-linear readout; `tanh(Σ pairs)` breaks it outside the small-signal regime, and budget renormalization further distorts relative pair strengths. Prescribed response: diagnose with the oracle arm at small fast / linear read before considering any I-1 revision (sign-off required).
  - **D-faint (magnitude):** α = 0.1 may leave even a correctly-written pair below the threshold to flip downstream argmax. Diagnose via oracle-arm α sweep (oracle arm only, as measurement); an I-1 change needs the evidence package and human sign-off.
  These are diagnosed — not guessed — by the oracle-write positive control (RULINGS_2 R6): perfect writes that still fail localize the gap to *read* (D-squash/D-faint); perfect writes that succeed localize it to *write selection* (D-smear).
- **Key contamination in oracle/write construction** (RULINGS_3, 2026-08-14): In-context forwards leak the answer into the query position via attention; representations recorded there are invalid as clean keys. Clean keys must come from pair-absent forwards. The first two oracle constructions (2026-08-14) stored bindings from (query + leaked answer)→answer and were therefore unreadable for reasons unrelated to the read path; their 0.000 does not by itself prove a retrieval gap. The decisive test (Oracle v3) uses a clean key from a pair-absent forward, writes into the last block's `ff_out` (output adds directly to the residual feeding `ln_f`→head), and sets the post-direction to the value token's tied unembedding row. This is a reusable lesson for any future write-construction.
- **Read failure is KEY NON-ORTHOGONALITY, not a dead locus** (RULINGS_3 V2 diagnostic, 2026-08-14): last-block ff_out clean keys for distinct held-out keys are ~91% collinear (mean |cos| 0.914, max 0.947), dominated by the shared query-slot/format direction. The outer-product read therefore blends all stored values (non-selective) and recall sits at chance. FIXABLE, not fundamental: the associative read retrieves at 1.000 with orthonormal keys, and whitening/decorrelating the EXISTING real keys restores 1.000 offline recall. Prescribed fix: a learned key projection (decorrelating map) before write and read, trainable by the meta-loop; do NOT abandon the additive fast-weight locus, do NOT relax I-1/I-2. Caveat: offline read isolates the associative math (no tanh/budget/residual/ln competition); confirm in-model after the projection is built.
- **Background � the read-organ is not a toy patch** (RULINGS PROGRAM_REVIEW record note): the 0.914 key-collinearity is transformer ANISOTROPY (a universal, scale-independent property of learned representations), so a decorrelating projection before storage is required at any scale, not just the toy. This organ independently re-derives two structures of the complementary-learning-systems (CLS) architecture: a separate fast-memory locus, and a decorrelating (dentate-gyrus-style pattern-separation) projection before storage. The projection should EXPAND + SPARSIFY, not merely rotate. This convergence-from-negative-results is the paper's spine regardless of where the four-in-one lands.
- **alpha*tanh(fast) is a DEAD RETRIEVAL PATH** (RULINGS DG_INTEGRATION_PINS �0, 2026-08-14): step 1 proved by controlled construction that a whitened key retrieving at 1.000 offline collapses to 0.000 through the model, because the elementwise tanh on the weight matrix destroys linear associative retrieval (D-squash, confirmed in-model). Standing consequence: the additive fast-weight read remains valid as bounded MODULATION of computation, but NO associative read may ever route through it, in this build or any future one. Associative retrieval must be linear end-to-end from code to logit (see DG module: separate linear M, read injected as a direct logit bias after the head).
