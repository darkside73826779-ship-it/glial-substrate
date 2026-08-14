"""DG smart auto-tuner (Rebecca): a feedback loop that drives DG's controls to
their best STABLE recall and runs until it can't improve.

Not a blind grid: coordinate ascent with restarts. Each candidate is scored as
  score = mean_recall - LAMBDA * std_recall   over R independent query resamples,
so the optimizer prefers a steady plateau (low variance) over a lucky spike.
Stops when no coordinate move beats the incumbent by > EPS for PATIENCE rounds.

Efficiency (API-frugal): pretrain ONCE, cache eval sets once; every candidate is
pure tensor post-processing on the cache. Streams phase="autotune" to
metrics.jsonl (monitor 'DG sweep' card shows the climb).

Knobs tuned: d_dg (expansion), k (sparsity), temp (completion softmax),
gamma (injection gain). P is random-frozen per d_dg (this is R2 tuning; R3
learned-P is a separate future step). Diagnostic; no learning run; invariants
unchanged.

Usage:  python scripts/dg_autotune.py [--fast]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from substrate import AssociativeRecall, SubstrateConfig, TinyTransformer
from substrate.dg import kwta, mean_abs_cos
from substrate.metrics import MetricsLog
from run_experiment import pretrain
from dg_inmodel import capture_key

LAMBDA = 0.5     # variance penalty (steady line preference)
EPS = 0.003      # min improvement to count as a move
PATIENCE = 2     # full no-improve passes before stopping


class Tuner:
    def __init__(self, model, ff_out, task, mapping, keys, device, args):
        self.m, self.ff, self.task, self.map = model, ff_out, task, mapping
        self.keys, self.dev, self.args = keys, device, args
        self.emb = model.tok.weight.detach()
        self.n_keys = task.n_keys
        self.store_vals = torch.stack([
            self.emb[task.n_keys + mapping[k]] /
            (self.emb[task.n_keys + mapping[k]].norm() + 1e-8) for k in keys])
        # store keys captured once (context A); query sets captured R times
        # from DIFFERENT contexts -> genuine retrieval + variance estimate.
        self.store_keys = torch.stack([
            capture_key(model, ff_out, task, mapping, keys, k, device, args,
                        100 + i) for i, k in enumerate(keys)])
        self.mu = self.store_keys.mean(0)
        self.qsets = []
        for r in range(args.resamples):
            qk = torch.stack([
                capture_key(model, ff_out, task, mapping, keys, k, device, args,
                            500 + r * 97 + i) for i, k in enumerate(keys)])
            self.qsets.append(qk)
        self.tgt = torch.arange(len(keys), device=device)
        self._Pcache = {}

    def _P(self, d_dg):
        if d_dg not in self._Pcache:
            g = torch.Generator().manual_seed(self.args.seed)
            self._Pcache[d_dg] = (torch.randn(d_dg, self.ff.in_features,
                                  generator=g) / (self.ff.in_features ** 0.5)
                                  ).to(self.dev)
        return self._Pcache[d_dg]

    @torch.no_grad()
    def score(self, cfg):
        d_dg, k, temp, gamma = cfg
        P = self._P(d_dg)
        sc = F.normalize(kwta((self.store_keys - self.mu) @ P.t(), k), dim=-1)
        overlap = mean_abs_cos(sc)
        recs = []
        for qk in self.qsets:
            qc = F.normalize(kwta((qk - self.mu) @ P.t(), k), dim=-1)
            att = torch.softmax((qc @ sc.t()) / temp, dim=1)
            yhat = att @ self.store_vals
            # value tokens argmax over the value vocab, matched to key index
            logits = yhat @ self.emb.t()
            pred_tok = logits.argmax(1)
            correct = torch.tensor(
                [self.n_keys + self.map[kk] for kk in self.keys],
                device=self.dev)
            recs.append((pred_tok == correct).float().mean().item())
        t = torch.tensor(recs)
        return t.mean().item(), t.std().item(), overlap


def coordinate_ascent(tuner, grids, start, mlog):
    cfg = dict(start)
    order = ["k", "temp", "gamma", "d_dg"]
    best_m, best_s, ov = tuner.score(tuple(cfg[x] for x in
                                     ["d_dg", "k", "temp", "gamma"]))
    best_score = best_m - LAMBDA * best_s
    step = 0
    print(f"start {cfg} -> mean {best_m:.3f} std {best_s:.3f}")
    no_improve = 0
    while no_improve < PATIENCE:
        improved = False
        for var in order:
            for val in grids[var]:
                if val == cfg[var]:
                    continue
                trial = dict(cfg); trial[var] = val
                m, s, ov = tuner.score(tuple(trial[x] for x in
                                       ["d_dg", "k", "temp", "gamma"]))
                sc = m - LAMBDA * s
                step += 1
                mlog.log(model="dg_autotune", phase="autotune", step=step,
                         d_dg=trial["d_dg"], k=trial["k"], temp=trial["temp"],
                         gamma=trial["gamma"], out_of_context_heldout=m,
                         recall_std=s, score=sc, best_dg=best_m,
                         dg_code_overlap=ov)
                if sc > best_score + EPS:
                    cfg, best_score, best_m, best_s = trial, sc, m, s
                    improved = True
                    print(f"  +move {var}={val}: mean {m:.3f} std {s:.3f} "
                          f"(score {sc:.3f})")
        no_improve = 0 if improved else no_improve + 1
    return cfg, best_m, best_s


@torch.no_grad()
def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    task = AssociativeRecall(n_keys=16, n_vals=16, pairs_per_seq=4, seed=args.seed)
    cfg = SubstrateConfig(group_size=4)
    chance = 1.0 / task.n_vals
    mlog = MetricsLog(Path(__file__).parent.parent / "metrics.jsonl",
                      run_name=f"dg-autotune-seed{args.seed}",
                      config={"seed": args.seed, "chance": chance,
                              "lambda": LAMBDA, "arm": "DG_autotuner"})
    torch.manual_seed(args.seed)
    model = TinyTransformer(task.vocab_size, d_model=args.d_model,
                            n_layers=args.layers, d_ff=args.d_ff, cfg=cfg,
                            use_substrate=True).to(device)
    print("--- pretrain (once) ---")
    with torch.enable_grad():
        pretrain(model, task, mapping_seed_range=10_000,
                 steps=args.pretrain_steps, device=device)
    ff_out = model.blocks[-1].ff_out
    mapping = task.random_mapping(task.held_keys, task.n_vals, 2)

    tuner = Tuner(model, ff_out, task, mapping, task.held_keys, device, args)
    grids = {"d_dg": args.dgs, "k": args.ks, "temp": args.temps,
             "gamma": args.gammas}
    # restarts from a few seeds in the grid, keep global best
    starts = [
        {"d_dg": args.dgs[len(args.dgs)//2], "k": 24, "temp": 0.03, "gamma": 20.0},
        {"d_dg": args.dgs[-1], "k": 32, "temp": 0.02, "gamma": 20.0},
        {"d_dg": args.dgs[0], "k": 16, "temp": 0.05, "gamma": 10.0},
    ]
    best = None
    for i, st in enumerate(starts):
        print(f"\n=== restart {i+1}/{len(starts)} ===")
        c, m, s = coordinate_ascent(tuner, grids, st, mlog)
        if best is None or m > best[1]:
            best = (c, m, s)
    c, m, s = best
    print("\n================ AUTOTUNE RESULT ================")
    print(f"best stable DG recall: mean {m:.3f}  std {s:.3f}  (chance {chance:.3f})")
    print(f"best config: {c}")
    print("steady plateau (low std) = set-and-forget viable; "
          "high std = needs an active controller.")
    out = Path(__file__).parent.parent / "dg_autotune_results.json"
    out.write_text(json.dumps({"seed": args.seed, "best_config": c,
                               "mean_recall": m, "std_recall": s,
                               "chance": chance}, indent=2))
    print(f"written: {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fast", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=256)
    p.add_argument("--pretrain_steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--ctx_batches", type=int, default=4)
    p.add_argument("--resamples", type=int, default=5)
    p.add_argument("--dgs", type=int, nargs="+", default=[2048, 4096, 8192])
    p.add_argument("--ks", type=int, nargs="+", default=[16, 24, 32, 48, 64, 96])
    p.add_argument("--temps", type=float, nargs="+",
                   default=[0.01, 0.02, 0.03, 0.05, 0.08])
    p.add_argument("--gammas", type=float, nargs="+", default=[10.0, 20.0, 40.0])
    a = p.parse_args()
    if a.fast:
        a.d_model, a.layers, a.d_ff = 64, 2, 128
        a.pretrain_steps = 300
        a.resamples = 3
        a.dgs = [1024, 2048]
        a.ks = [16, 32]
    run(a)


if __name__ == "__main__":
    main()
