"""Meta-training loop for the plasticity gates (DESIGN.md open work #1).

This is the outer loop AGENT_ORDERS.md §1 asks for. It is written entirely
AROUND the proven machinery in substrate/ — it does NOT modify layer.py,
model.py mode logic, the invariant tests, or the benchmark definitions.
The differentiable inner path it uses (functional_forward / functional_tick)
already exists and is verified by invariant I-4d.

Per episode (the shape mandated by §1):
  1. Sample a novel key->value mapping on the HELD-OUT pool.
  2. Initialize episode-local fast/trace/consol tensors (zeros, per layer).
     These are ordinary tensors passed by value — never the module buffers.
  3. Exposure phase: run sequences through functional_forward(x, fast),
     accumulate traces functionally (mirror of _accumulate_trace on local
     tensors, kept differentiable), and apply functional_tick at intervals
     to produce updated fast tensors.
  4. Probe phase: OUT-OF-CONTEXT queries (the pair is absent from the
     sequence) through functional_forward with the final fast tensors.
  5. Backprop probe loss to GateNet parameters and log_plastic_lr ONLY.

Gradient firewall self-guard (§1): after each step we assert at least one
GateNet parameter received a nonzero gradient. A stray .detach() anywhere on
the functional path would silently sever gradients to GateNet and produce a
trainer that runs flawlessly and learns nothing; the invariant suite tests
the path, not our USE of it, so this assertion is our responsibility and it
stays permanently.

Usage:
  python scripts/meta_train.py                 # full: multi-seed, §2 report
  python scripts/meta_train.py --fast          # smoke-scale
  python scripts/meta_train.py --seeds 0 1 2   # explicit seeds
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # import sibling script

from substrate import (AssociativeRecall, SubstrateConfig,
                       SubstrateController, SubstrateLinear, TinyTransformer)
from substrate.metrics import MetricsLog
from run_experiment import loss_fn, pretrain, recall_acc, runtime_exposure


# --------------------------------------------------------------------------- #
# Differentiable full-model forward using episode-local fast tensors.
#
# TinyTransformer.forward() calls each SubstrateLinear.forward(x), which reads
# the module's *buffer* fast (gradient-free by design, I-4). For the meta inner
# loop we must instead route those layers through functional_forward with our
# episode-local fast tensors so autograd can reach GateNet. We do this WITHOUT
# editing model.py: a context manager temporarily swaps each substrate layer's
# bound forward, then restores it. No project file changes; pure orchestration.
# --------------------------------------------------------------------------- #

@contextlib.contextmanager
def functional_substrate(layers, state):
    """Temporarily make each SubstrateLinear route through functional_forward
    using state[layer]['fast'], capturing its (x, y) activations for the trace
    mirror. Restores original forwards on exit."""
    saved = {}
    try:
        for lay in layers:
            saved[lay] = lay.forward

            def make_fwd(L):
                def fwd(x):
                    y = L.functional_forward(x, state[L]["fast"])
                    st = state[L]
                    st["x"], st["y"] = x, y
                    return y
                return fwd

            lay.forward = make_fwd(lay)
        yield
    finally:
        for lay, fwd in saved.items():
            lay.forward = fwd


def _trace_mirror(trace, x, y, in_features, out_features, decay):
    """Differentiable mirror of SubstrateLinear._accumulate_trace: EMA of
    outer(post, pre) with per-feature L2 normalization. No .detach() — this
    is inside the meta graph on purpose (§1: 'differentiably where needed')."""
    xf = F.normalize(x.reshape(-1, in_features), dim=-1)
    yf = F.normalize(y.reshape(-1, out_features), dim=-1)
    outer = yf.t() @ xf / max(xf.shape[0], 1)
    return decay * trace + (1.0 - decay) * outer


def init_episode_state(layers):
    """Zeros for fast/trace/consol per layer — ordinary tensors, never buffers."""
    state = {}
    for lay in layers:
        shape = (lay.out_features, lay.in_features)
        dev = lay.W_base.device
        state[lay] = {
            "fast": torch.zeros(shape, device=dev),
            "trace": torch.zeros(shape, device=dev),
            "consol": torch.zeros(shape, device=dev),
        }
    return state


def run_episode(model, layers, task, mapping, cfg, args, device):
    """One meta episode. Returns (probe_loss, gate_mean_scalar)."""
    state = init_episode_state(layers)
    gate_running = []

    with functional_substrate(layers, state):
        # --- exposure: stream held-out-pool sequences, tick at intervals ---
        for step in range(args.episode_exposure):
            X, T, _ = task.make_batch(args.batch, mapping, task.held_keys, device)
            _ = model(X)  # routes through functional_forward; caches x,y per layer
            for lay in layers:
                st = state[lay]
                st["trace"] = _trace_mirror(
                    st["trace"], st["x"], st["y"],
                    lay.in_features, lay.out_features, cfg.trace_decay)
            if (step + 1) % args.tick_every == 0:
                for lay in layers:
                    st = state[lay]
                    new_fast, gate = lay.functional_tick(
                        st["fast"], st["trace"], st["consol"], args.neuromod)
                    st["fast"] = new_fast
                    st["trace"] = st["trace"] * 0.5  # mirror tick's trace decay
                    gate_running.append(gate.mean())

        # --- probe: OUT-OF-CONTEXT query with final fast tensors ---
        Xp, Tp, _ = task.make_batch(
            args.batch, mapping, task.held_keys, device,
            include_query_pair=False)
        logits = model(Xp)  # still functional (final fast in state)
    probe_loss = loss_fn(logits, Tp)
    gate_mean = (torch.stack(gate_running).mean() if gate_running
                 else torch.tensor(0.0))
    return probe_loss, gate_mean


def meta_train(model, task, cfg, args, device, mlog=None, seed=0):
    """Outer loop: shape GateNet + log_plastic_lr with backprop through the
    functional inner path. Runtime buffers are never touched here."""
    model.set_mode("meta")  # gate_net + log_plastic_lr require grad; W_base frozen
    layers = model.substrate_layers()

    # §3 rung 1: optional GateNet final-bias raise (gates start mostly-off by
    # design). Applied from the script via the tunable surface, not by editing
    # layer.py. Left at the layer default unless --gate_bias is given.
    if args.gate_bias is not None:
        with torch.no_grad():
            for lay in layers:
                lay.gate_net.net[-1].bias.fill_(args.gate_bias)

    meta_params, names = [], []
    for n, p in model.named_parameters():
        if p.requires_grad and ("gate_net" in n or "log_plastic_lr" in n):
            meta_params.append(p)
            names.append(n)
    assert meta_params, "no meta parameters require grad — set_mode('meta') failed"
    gate_params = [p for n, p in zip(names, meta_params) if "gate_net" in n]

    opt = torch.optim.Adam(meta_params, lr=args.meta_lr)

    for ep in range(args.episodes):
        mapping = task.random_mapping(task.held_keys, task.n_vals,
                                      seed=1000 + ep)
        probe_loss, gate_mean = run_episode(
            model, layers, task, mapping, cfg, args, device)

        opt.zero_grad()
        probe_loss.backward()

        # --- gradient firewall self-guard (§1). Permanent. ---
        gnorm = 0.0
        for p in gate_params:
            if p.grad is not None:
                gnorm += float(p.grad.detach().pow(2).sum())
        gnorm = gnorm ** 0.5
        if gnorm == 0.0:
            raise RuntimeError(
                "GateNet received ZERO gradient — the functional path is "
                "severed (likely a stray .detach()). Refusing to train a "
                "loop that cannot learn (AGENT_ORDERS.md §1).")

        opt.step()

        if ep % max(args.episodes // 20, 1) == 0 or ep == args.episodes - 1:
            print(f"  [seed {seed}] episode {ep:4d}  probe_loss "
                  f"{probe_loss.item():.4f}  gate {gate_mean.item():.3f}  "
                  f"gatenet_gradnorm {gnorm:.4e}")
        if mlog is not None:
            # New scalars from the meta loop — explicitly allowed by
            # MONITOR_RULES.md ('probe accuracy per episode, GateNet gradient
            # norm, episode wall-clock'). Values already exist here; plain
            # floats only; no hooks, no retained graph.
            mlog.log(model="substrate", phase="meta", episode=ep,
                     probe_loss=probe_loss.item(), gate_mean=gate_mean.item(),
                     gatenet_gradnorm=gnorm)
    return model


def deploy_eval(model, task, args, device, substrate: bool,
                mlog=None, model_name=""):
    """Real deployment path (NOT the functional inner loop): reset substrate,
    run gradient-free runtime exposure with ticks, then probe. This is what
    the meta-trained gates actually produce at runtime."""
    eval_mapping = task.random_mapping(task.held_keys, task.n_vals, seed=2)
    pre_mapping = task.random_mapping(task.train_keys, task.n_vals, seed=1)
    if substrate:
        model.reset_substrate()
        model.set_mode("runtime")

    before = {
        "in_context_pretrain_pool": recall_acc(
            model, task, pre_mapping, task.train_keys, device=device),
        "in_context_heldout_pool": recall_acc(
            model, task, eval_mapping, task.held_keys, device=device),
        "out_of_context_heldout": recall_acc(
            model, task, eval_mapping, task.held_keys,
            include_query_pair=False, device=device),
    }
    if mlog is not None:
        mlog.phase(f"{model_name}:probe-before")
        mlog.log(model=model_name, phase="probe", when="before", **before)
        mlog.phase(f"{model_name}:exposure")
    runtime_exposure(model, task, eval_mapping, args.exposure_steps, device,
                     substrate=substrate, tick_every=args.tick_every,
                     mlog=mlog, model_name=model_name)
    after = {
        "in_context_pretrain_pool": recall_acc(
            model, task, pre_mapping, task.train_keys, device=device),
        "in_context_heldout_pool": recall_acc(
            model, task, eval_mapping, task.held_keys, device=device),
        "out_of_context_heldout": recall_acc(
            model, task, eval_mapping, task.held_keys,
            include_query_pair=False, device=device),
    }
    if mlog is not None:
        mlog.log(model=model_name, phase="probe", when="after", **after)
    return before, after


def run_seed(seed, args, device, mlog):
    print(f"\n================ seed {seed} ================")
    task = AssociativeRecall(n_keys=16, n_vals=16, pairs_per_seq=4, seed=seed)
    cfg = SubstrateConfig(group_size=4, plastic_lr=args.plastic_lr)

    # --- baseline: parity model, identical init, no meta (the fair frame) ---
    torch.manual_seed(seed)
    base = TinyTransformer(task.vocab_size, d_model=args.d_model,
                           n_layers=args.layers, d_ff=args.d_ff, cfg=cfg,
                           use_substrate=False).to(device)
    print("--- baseline pretrain ---")
    if mlog is not None:
        mlog.phase("baseline:pretrain")
    pretrain(base, task, mapping_seed_range=10_000,
             steps=args.pretrain_steps, device=device,
             mlog=mlog, model_name="baseline")
    b_before, b_after = deploy_eval(base, task, args, device, substrate=False,
                                    mlog=mlog, model_name="baseline")

    # --- substrate: identical init, pretrain, then META-TRAIN the gates ---
    torch.manual_seed(seed)
    model = TinyTransformer(task.vocab_size, d_model=args.d_model,
                            n_layers=args.layers, d_ff=args.d_ff, cfg=cfg,
                            use_substrate=True).to(device)
    print("--- substrate pretrain ---")
    if mlog is not None:
        mlog.phase("substrate:pretrain")
    pretrain(model, task, mapping_seed_range=10_000,
             steps=args.pretrain_steps, device=device,
             mlog=mlog, model_name="substrate")

    print("--- meta-training gates ---")
    t0 = time.time()
    meta_train(model, task, cfg, args, device, mlog=mlog, seed=seed)
    meta_secs = round(time.time() - t0, 1)

    s_before, s_after = deploy_eval(model, task, args, device, substrate=True,
                                    mlog=mlog, model_name="substrate")

    return {
        "seed": seed,
        "baseline": {"before": b_before, "after": b_after},
        "substrate": {"before": s_before, "after": s_after},
        "meta_seconds": meta_secs,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fast", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=256)
    p.add_argument("--pretrain_steps", type=int, default=3000)
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--episode_exposure", type=int, default=24)
    p.add_argument("--tick_every", type=int, default=4)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--exposure_steps", type=int, default=800)
    p.add_argument("--meta_lr", type=float, default=1e-3)
    # §3 ladder knobs (use in the mandated order; do NOT touch alpha/budget):
    p.add_argument("--gate_bias", type=float, default=None,  # rung 1
                   help="override GateNet final-layer bias init (gates start off)")
    p.add_argument("--neuromod", type=float, default=1.0,    # rung 2 signal
                   help="broadcast neuromodulator during meta exposure")
    p.add_argument("--plastic_lr", type=float, default=0.5)  # rung 4
    a = p.parse_args()
    if a.fast:
        a.seeds = a.seeds[:1]
        a.d_model, a.layers, a.d_ff = 64, 2, 128
        a.pretrain_steps = 300
        a.episodes = 40
        a.episode_exposure = 12
        a.exposure_steps = 120

    device = "cuda" if torch.cuda.is_available() else "cpu"
    chance = 1.0 / 16
    mlog = MetricsLog(Path(__file__).parent.parent / "metrics.jsonl",
                      run_name="meta-train",
                      config={"seeds": a.seeds, "episodes": a.episodes,
                              "chance": chance, "gate_bias": a.gate_bias,
                              "neuromod": a.neuromod, "plastic_lr": a.plastic_lr})

    results = [run_seed(s, a, device, mlog) for s in a.seeds]

    # -------- §2 report: the conjunction, honest numbers, per seed -------- #
    print("\n================ §2 report ================")
    print(f"chance = {chance:.4f}")
    print(f"{'seed':>4} | {'model':10s} | {'oo-ctx before':>13} | "
          f"{'oo-ctx after':>12} | {'oo d':>7} | {'in-ctx held d':>13} | "
          f"{'pretrain-pool (forget)':>22}")
    for r in results:
        for m in ("baseline", "substrate"):
            b, af = r[m]["before"], r[m]["after"]
            oo_d = af["out_of_context_heldout"] - b["out_of_context_heldout"]
            ic_d = af["in_context_heldout_pool"] - b["in_context_heldout_pool"]
            forget = (b["in_context_pretrain_pool"]
                      - af["in_context_pretrain_pool"])
            print(f"{r['seed']:>4} | {m:10s} | "
                  f"{b['out_of_context_heldout']:>13.3f} | "
                  f"{af['out_of_context_heldout']:>12.3f} | {oo_d:>+7.3f} | "
                  f"{ic_d:>+13.3f} | {forget:>+22.3f}")

    # Done-check (§2 conjunction), reported honestly — a clean negative is a
    # valid, reportable outcome; a contaminated positive is not.
    margins = [r["substrate"]["after"]["out_of_context_heldout"]
               for r in results]
    over_base = [r["substrate"]["after"]["out_of_context_heldout"]
                 - r["baseline"]["after"]["out_of_context_heldout"]
                 for r in results]
    above_chance_all = all(m > chance + 0.02 for m in margins)
    beats_base_all = all(d > 0 for d in over_base)
    print("\n§2 clause 1 (effect): substrate oo-ctx above chance across ALL "
          f"seeds AND beats baseline: {above_chance_all and beats_base_all}")
    print(f"    per-seed substrate oo-ctx: {[round(m,3) for m in margins]} "
          f"(chance {chance:.3f})")
    print(f"    per-seed (substrate - baseline) oo-ctx: "
          f"{[round(d,3) for d in over_base]}")
    print("§2 clause 2 (contract): run scripts/test_invariants.py separately "
          "— must print 'all invariants hold'.")
    print("§2 clause 3 (clean comparison): baseline numbers must match the "
          "no-meta clean-state baseline within seed noise (baseline code "
          "untouched by this script).")

    out = Path(__file__).parent.parent / "meta_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwritten: {out}")
    print("wall-clock per meta run (per seed): "
          f"{[r['meta_seconds'] for r in results]} s")


if __name__ == "__main__":
    main()
