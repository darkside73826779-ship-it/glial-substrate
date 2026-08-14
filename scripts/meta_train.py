"""Meta-training loop for the plasticity gates (DESIGN.md open work #1).

VERSION 2 — implements RULINGS_META_TRAIN.md D1+D2+D3 (the designing
assistant's sanctioned corrections to the v1 negative run). v1 is preserved
verbatim as scripts/meta_train_v1_negative.py for provenance.

This is the outer loop AGENT_ORDERS.md §1 asks for, written around the proven
machinery in substrate/. It does NOT modify the runtime buffer discipline
(I-4, I-6) or the invariant tests' meaning. The only sanctioned core change —
a first-class functional forward path in model.py, guarded by new invariant
I-11 — was authorized by the rulings (D1).

Corrections vs v1:
  D1  No monkey-patching. The full transformer is run functionally via
      model(idx, fast_map=..., act_out=...): fast_map threads episode-local
      fast tensors to functional_forward; act_out captures per-layer (x, y)
      for the differentiable trace mirror. Runtime buffers untouched.
  D2  Surprise-driven neuromodulator (SubstrateController's formula, computed
      functionally from episode losses), NOT a constant. Episodes contain
      BOTH regimes so the gate has neuromod contrast to condition on:
        - write-worthy: novel held-out pairings, high surprise;
        - write-hostile: familiar pretrain-pool sequences, low surprise.
      GateNet negative bias init stands; we do not raise it preemptively.
  D3  Retention term. L = L_recall + lambda_ret * L_retention, where
      L_retention penalizes degradation of pretrain-pool recall probed through
      the functional path with the episode's final fast tensors. consol is the
      organ meant to be trained against forgetting; v1 left it unused.

Gradient firewall self-guard (§1): after each step we assert at least one
GateNet parameter received a nonzero gradient — permanent.

Success (RULINGS §4, tightened §2 clause 1): substrate out-of-context recall
above chance across ALL 3 seeds AND beating baseline, WITH pretrain-pool
forgetting < 0.10. A clean negative under the §3 ladder is equally reportable.

Usage:
  python scripts/meta_train.py                 # full: 3 seeds
  python scripts/meta_train.py --fast          # smoke-scale
  python scripts/meta_train.py --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from substrate import (AssociativeRecall, SubstrateConfig, TinyTransformer)
from substrate.metrics import MetricsLog
from run_experiment import loss_fn, pretrain, recall_acc, runtime_exposure


FORGET_BOUND = 0.10  # RULINGS §4 stopping-rule forgetting ceiling


# --------------------------------------------------------------------------- #
# episode state + differentiable trace mirror                                 #
# --------------------------------------------------------------------------- #

def init_episode_state(layers):
    """Fresh zero fast/trace/consol per layer — ordinary tensors, never buffers."""
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


def _trace_mirror(trace, x, y, in_f, out_f, decay):
    """Differentiable mirror of SubstrateLinear._accumulate_trace (kept in the
    meta graph on purpose; §1 'differentiably where needed')."""
    xf = F.normalize(x.reshape(-1, in_f), dim=-1)
    yf = F.normalize(y.reshape(-1, out_f), dim=-1)
    outer = yf.t() @ xf / max(xf.shape[0], 1)
    return decay * trace + (1.0 - decay) * outer


class SurpriseEMA:
    """Functional mirror of SubstrateController.neuromod(): a scalar in (0,1)
    that rises when recent loss is surprising/improving vs the slow baseline.
    Computed from episode losses so meta-training and runtime deployment share
    the same neuromod semantics (RULINGS D2)."""

    def __init__(self, k=8.0):
        self.k = k
        self.fast = None
        self.slow = None

    def observe(self, loss_val: float) -> float:
        self.fast = loss_val if self.fast is None else 0.7 * self.fast + 0.3 * loss_val
        self.slow = loss_val if self.slow is None else 0.99 * self.slow + 0.01 * loss_val
        if self.fast is None or self.slow is None:
            return 0.5
        rel = (self.slow - self.fast) / (abs(self.slow) + 1e-8)
        return float(torch.sigmoid(torch.tensor(self.k * rel)))


# --------------------------------------------------------------------------- #
# one meta episode                                                            #
# --------------------------------------------------------------------------- #

def run_episode(model, layers, task, novel_map, pre_map, cfg, args, device,
                mlog=None, ep=0):
    """Exposure with write-worthy (novel) / write-hostile (familiar) contrast,
    surprise-driven neuromod, ticks on the functional path; then recall +
    retention probes. Returns component losses and telemetry."""
    state = init_episode_state(layers)
    surprise = SurpriseEMA(k=args.neuromod_gain)
    gate_running, neuromod_running = [], []

    g = torch.Generator(device="cpu").manual_seed(10_000 + ep)
    for step in range(args.episode_exposure):
        write_worthy = bool(torch.rand(1, generator=g).item() >= args.hostile_prob)
        if write_worthy:
            X, T, _ = task.make_batch(args.batch, novel_map, task.held_keys, device)
        else:
            X, T, _ = task.make_batch(args.batch, pre_map, task.train_keys, device)

        act = {}
        fast_map = {lay: state[lay]["fast"] for lay in layers}
        logits = model(X, fast_map=fast_map, act_out=act)
        step_loss = loss_fn(logits, T)
        neuromod = surprise.observe(step_loss.item())  # scalar, non-differentiable
        neuromod_running.append(neuromod)

        for lay in layers:
            x, y = act[lay]
            st = state[lay]
            st["trace"] = _trace_mirror(st["trace"], x, y,
                                        lay.in_features, lay.out_features,
                                        cfg.trace_decay)

        if (step + 1) % args.tick_every == 0:
            for lay in layers:
                st = state[lay]
                new_fast, gate = lay.functional_tick(
                    st["fast"], st["trace"], st["consol"], neuromod)
                st["fast"] = new_fast
                st["trace"] = st["trace"] * 0.5  # mirror tick's trace decay
                gate_running.append(gate.mean())

    final_fast = {lay: state[lay]["fast"] for lay in layers}

    # L_recall: out-of-context recall on the novel pool written this episode.
    Xr, Tr, _ = task.make_batch(args.batch, novel_map, task.held_keys, device,
                                include_query_pair=False)
    L_recall = loss_fn(model(Xr, fast_map=final_fast), Tr)

    # L_retention (D3): pretrain-pool recall must survive the episode's writes.
    # In-context on train_keys, through the SAME final fast tensors: if the
    # gates wrote indiscriminately over pretrain knowledge, this loss rises.
    Xk, Tk, _ = task.make_batch(args.batch, pre_map, task.train_keys, device)
    L_retention = loss_fn(model(Xk, fast_map=final_fast), Tk)

    gate_mean = (torch.stack(gate_running).mean() if gate_running
                 else torch.tensor(0.0, device=device))
    neuromod_mean = sum(neuromod_running) / max(len(neuromod_running), 1)
    return L_recall, L_retention, gate_mean, neuromod_mean


# --------------------------------------------------------------------------- #
# outer loop                                                                  #
# --------------------------------------------------------------------------- #

def meta_train(model, task, cfg, args, device, mlog=None, seed=0):
    model.set_mode("meta")  # gate_net + log_plastic_lr require grad; W_base frozen
    layers = model.substrate_layers()

    if args.gate_bias is not None:  # §3 rung 1 (default: leave layer's -1.0)
        with torch.no_grad():
            for lay in layers:
                lay.gate_net.net[-1].bias.fill_(args.gate_bias)

    names = [n for n, p in model.named_parameters()
             if p.requires_grad and ("gate_net" in n or "log_plastic_lr" in n)]
    meta_params = [p for n, p in model.named_parameters() if n in names]
    gate_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "gate_net" in n]
    assert meta_params, "no meta parameters require grad — set_mode('meta') failed"
    opt = torch.optim.Adam(meta_params, lr=args.meta_lr)

    pre_map = task.random_mapping(task.train_keys, task.n_vals, seed=1)

    for ep in range(args.episodes):
        novel_map = task.random_mapping(task.held_keys, task.n_vals, seed=1000 + ep)
        L_recall, L_ret, gate_mean, neuromod_mean = run_episode(
            model, layers, task, novel_map, pre_map, cfg, args, device,
            mlog=mlog, ep=ep)
        loss = L_recall + args.lambda_ret * L_ret

        opt.zero_grad()
        loss.backward()

        gnorm = sum(float(p.grad.detach().pow(2).sum())
                    for p in gate_params if p.grad is not None) ** 0.5
        if gnorm == 0.0:
            raise RuntimeError(
                "GateNet received ZERO gradient — functional path severed "
                "(likely a stray .detach()). Refusing to train a loop that "
                "cannot learn (AGENT_ORDERS.md §1).")
        opt.step()

        if ep % max(args.episodes // 20, 1) == 0 or ep == args.episodes - 1:
            print(f"  [seed {seed}] ep {ep:4d}  L_recall {L_recall.item():.3f}  "
                  f"L_ret {L_ret.item():.3f}  gate {gate_mean.item():.3f}  "
                  f"neuromod {neuromod_mean:.3f}  gradnorm {gnorm:.2e}")
        if mlog is not None:
            mlog.log(model="substrate", phase="meta", episode=ep,
                     probe_loss=L_recall.item(), retention_loss=L_ret.item(),
                     gate_mean=gate_mean.item(), neuromod=neuromod_mean,
                     gatenet_gradnorm=gnorm)
    return model


def deploy_eval(model, task, args, device, substrate: bool,
                mlog=None, model_name=""):
    """Real gradient-free deployment path (SubstrateController surprise neuromod)."""
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

    torch.manual_seed(seed)
    base = TinyTransformer(task.vocab_size, d_model=args.d_model,
                           n_layers=args.layers, d_ff=args.d_ff, cfg=cfg,
                           use_substrate=False).to(device)
    print("--- baseline pretrain ---")
    if mlog is not None:
        mlog.phase("baseline:pretrain")
    pretrain(base, task, mapping_seed_range=10_000, steps=args.pretrain_steps,
             device=device, mlog=mlog, model_name="baseline")
    b_before, b_after = deploy_eval(base, task, args, device, substrate=False,
                                    mlog=mlog, model_name="baseline")

    torch.manual_seed(seed)
    model = TinyTransformer(task.vocab_size, d_model=args.d_model,
                            n_layers=args.layers, d_ff=args.d_ff, cfg=cfg,
                            use_substrate=True).to(device)
    print("--- substrate pretrain ---")
    if mlog is not None:
        mlog.phase("substrate:pretrain")
    pretrain(model, task, mapping_seed_range=10_000, steps=args.pretrain_steps,
             device=device, mlog=mlog, model_name="substrate")

    print("--- meta-training gates (D2 surprise neuromod + D3 retention) ---")
    t0 = time.time()
    meta_train(model, task, cfg, args, device, mlog=mlog, seed=seed)
    meta_secs = round(time.time() - t0, 1)

    s_before, s_after = deploy_eval(model, task, args, device, substrate=True,
                                    mlog=mlog, model_name="substrate")
    return {"seed": seed,
            "baseline": {"before": b_before, "after": b_after},
            "substrate": {"before": s_before, "after": s_after},
            "meta_seconds": meta_secs}


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
    p.add_argument("--hostile_prob", type=float, default=0.5,
                   help="fraction of exposure steps that are write-hostile (D2)")
    p.add_argument("--lambda_ret", type=float, default=1.0,
                   help="retention-term weight (D3); log every change")
    # §3 ladder knobs (mandated order; alpha/budget_frac deliberately absent):
    p.add_argument("--gate_bias", type=float, default=None)   # rung 1
    p.add_argument("--neuromod_gain", type=float, default=8.0)  # rung 2
    p.add_argument("--plastic_lr", type=float, default=0.5)   # rung 4
    a = p.parse_args()
    if a.fast:
        a.seeds = a.seeds[:1]
        a.d_model, a.layers, a.d_ff = 64, 2, 128
        a.pretrain_steps = 300
        a.episodes = 60
        a.episode_exposure = 16
        a.exposure_steps = 120

    device = "cuda" if torch.cuda.is_available() else "cpu"
    chance = 1.0 / 16
    mlog = MetricsLog(Path(__file__).parent.parent / "metrics.jsonl",
                      run_name="meta-train-v2",
                      config={"seeds": a.seeds, "episodes": a.episodes,
                              "chance": chance, "gate_bias": a.gate_bias,
                              "neuromod_gain": a.neuromod_gain,
                              "hostile_prob": a.hostile_prob,
                              "lambda_ret": a.lambda_ret,
                              "plastic_lr": a.plastic_lr,
                              "forget_bound": FORGET_BOUND})

    results = [run_seed(s, a, device, mlog) for s in a.seeds]

    print("\n================ §2 report (v2, RULINGS §4 success bar) ========")
    print(f"chance = {chance:.4f}   forgetting bound = {FORGET_BOUND}")
    print(f"{'seed':>4} | {'model':10s} | {'oo before':>9} | {'oo after':>8} | "
          f"{'oo d':>7} | {'in-ctx held d':>13} | {'pretrain forget':>15}")
    for r in results:
        for m in ("baseline", "substrate"):
            b, af = r[m]["before"], r[m]["after"]
            oo_d = af["out_of_context_heldout"] - b["out_of_context_heldout"]
            ic_d = af["in_context_heldout_pool"] - b["in_context_heldout_pool"]
            forget = b["in_context_pretrain_pool"] - af["in_context_pretrain_pool"]
            print(f"{r['seed']:>4} | {m:10s} | {b['out_of_context_heldout']:>9.3f} "
                  f"| {af['out_of_context_heldout']:>8.3f} | {oo_d:>+7.3f} | "
                  f"{ic_d:>+13.3f} | {forget:>+15.3f}")

    oo = [r["substrate"]["after"]["out_of_context_heldout"] for r in results]
    over_base = [r["substrate"]["after"]["out_of_context_heldout"]
                 - r["baseline"]["after"]["out_of_context_heldout"] for r in results]
    forget = [r["substrate"]["before"]["in_context_pretrain_pool"]
              - r["substrate"]["after"]["in_context_pretrain_pool"] for r in results]
    effect = all(m > chance + 0.02 for m in oo) and all(d > 0 for d in over_base)
    retained = all(f < FORGET_BOUND for f in forget)
    print("\nRULINGS §4 success (effect AND forgetting<0.10 across all seeds): "
          f"{effect and retained}")
    print(f"    substrate oo-ctx per seed: {[round(x,3) for x in oo]} "
          f"(chance {chance:.3f})")
    print(f"    (substrate - baseline) oo-ctx: {[round(x,3) for x in over_base]}")
    print(f"    substrate forgetting per seed: {[round(x,3) for x in forget]} "
          f"(bound {FORGET_BOUND})")
    print("    §2 clause 2 (contract): run scripts/test_invariants.py — must "
          "print 'all invariants hold' (now includes I-11).")

    out = Path(__file__).parent.parent / "meta_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwritten: {out}")
    print(f"wall-clock per seed: {[r['meta_seconds'] for r in results]} s")


if __name__ == "__main__":
    main()
