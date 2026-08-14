"""Oracle v3 — the decisive retrieval test (RULINGS_3 F1/F2/F3).

Fixes the key-contamination flaw RULINGS_3 charged to the R6 spec: in an
in-context-correct forward, attention has already routed the answer into the
query position before the FFN sees it, so v1/v2 stored (query+leaked answer)->
answer. v3 removes every such artifact by MAXIMAL-DIRECTNESS construction:

  1. CLEAN KEY: the query-position pre-activation (input to the last block's
     ff_out) recorded from a PAIR-ABSENT forward — exactly the out-of-context
     condition the probe presents. No answer can leak in.
  2. LOCUS: the last block's ff_out ONLY. Its output adds directly to the
     residual stream feeding ln_f -> head. Shortest path perturbation->logits.
  3. POST = the value token's tied embedding row (the unembedding direction),
     so the write aims signal straight at the target logit.

Then a DOSE-RESPONSE grid (RULINGS_3 F2), diagnostic-only, bounds restored
after every cell, never a learning run:
  budget_frac in {0.25, 0.5, 1.0, off} x alpha in {0.1, 0.4, 1.0}
For each cell log: recall, DELIVERED DOSE ||W_eff-W_base||/||W_base|| at the
written rows, and the target-token LOGIT-MARGIN SHIFT at the probe position.
Plus a matched-magnitude LINEAR-READ cell (tanh->identity) closing D-squash.

Verdict:
  any cell clears chance decisively -> additive bounded-perturbation READ works;
    earlier 0.000 were construction artifacts. Retrieval locus validated.
  no cell moves recall at any delivered dose -> additive perturbation of
    pretrained weights is dead as a memory locus (RULINGS_3 F4 branch).

Usage:  python scripts/oracle_probe_v3.py [--fast]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from substrate import (AssociativeRecall, SubstrateConfig, SubstrateLinear,
                       TinyTransformer)
from substrate.metrics import MetricsLog
from run_experiment import pretrain, recall_acc


def pair_absent_batch(task, mapping, keys, k, device, batch, n_pairs):
    """Sequences that do NOT contain the pair (k, val): random filler pairs,
    then [SEP, QUERY, k]. The final token position is the clean query on k,
    exactly the out-of-context probe condition. Uniform length -> final pos
    is last index."""
    seqs = []
    others = [x for x in keys if x != k]
    for _ in range(batch):
        fill = torch.tensor(others)[torch.randperm(len(others))][
            : n_pairs].tolist()
        seq = []
        for kk in fill:
            seq += [kk, task.n_keys + mapping[kk]]
        seq += [task.SEP, task.QUERY, k]
        seqs.append(torch.tensor(seq))
    X = torch.stack(seqs).to(device)
    return X, X.shape[1] - 1


@torch.no_grad()
def clean_keys(model, ff_out, task, mapping, keys, device, args):
    """Clean key per held key k: mean pair-absent query-position pre-activation
    (input to ff_out) — the d_ff vector the probe will actually present."""
    layers = model.substrate_layers()
    keyvec = {}
    for k in keys:
        acc = None
        for _ in range(args.oracle_batches):
            X, fpos = pair_absent_batch(task, mapping, keys, k, device,
                                        args.batch, task.pairs_per_seq)
            act = {}
            fmap = {l: torch.zeros(l.out_features, l.in_features, device=device)
                    for l in layers}
            _ = model(X, fast_map=fmap, act_out=act)
            pre, _post = act[ff_out]                 # pre = g (B,T,d_ff)
            v = pre[:, fpos, :].mean(0)              # (d_ff,)
            acc = v if acc is None else acc + v
        keyvec[k] = acc / args.oracle_batches
    return keyvec


@torch.no_grad()
def build_M(model, ff_out, task, mapping, keys, keyvec, device):
    """Associative matrix into ff_out.fast (d_model x d_ff):
    sum_k outer(emb[value_k], keyhat_k). emb = tied unembedding row."""
    emb = model.tok.weight                            # (vocab, d_model), tied head
    M = torch.zeros(ff_out.out_features, ff_out.in_features, device=device)
    for k in keys:
        vt = task.n_keys + mapping[k]
        p = keyvec[k] / (keyvec[k].norm() + 1e-8)     # (d_ff,)
        q = emb[vt] / (emb[vt].norm() + 1e-8)         # (d_model,)
        M += torch.outer(q, p)
    return M / max(len(keys), 1)


@torch.no_grad()
def budget_rescale(ff_out, fast, budget_frac, alpha):
    """Per-group budget on ff_out's own W_base scale; budget_frac=None => off."""
    if budget_frac is None:
        return fast
    of, ng, gs = ff_out.out_features, ff_out.n_groups, ff_out.cfg.group_size
    contrib = alpha * torch.tanh(fast)
    gnorm = contrib.view(of, ng, gs).norm(dim=-1)
    bnorm = ff_out.W_base.view(of, ng, gs).norm(dim=-1)
    limit = budget_frac * bnorm + 1e-8
    scale = torch.clamp(limit / (gnorm + 1e-8), max=1.0)
    return (fast.view(of, ng, gs) * scale.unsqueeze(-1)).reshape_as(fast)


def _linear_eff_weight(self, fast=None):
    f = self.fast if fast is None else fast
    return self.W_base + self.cfg.alpha * f           # identity, not tanh


@torch.no_grad()
def measure_cell(model, ff_out, task, mapping, keys, M, device, args,
                 budget_frac, alpha, linear=False):
    """Install M at (budget_frac, alpha), measure recall + delivered dose +
    logit-margin shift, restore. Fully diagnostic."""
    layers = model.substrate_layers()
    saved_alpha = ff_out.cfg.alpha
    saved_eff = ff_out.effective_weight
    ff_out.cfg.alpha = alpha
    if linear:
        ff_out.effective_weight = _linear_eff_weight.__get__(ff_out, SubstrateLinear)

    fast = budget_rescale(ff_out, M.clone(), budget_frac, alpha)
    model.set_mode("runtime")
    for l in layers:
        l.reset_substrate()
    ff_out.fast.copy_(fast)

    # delivered dose: ||W_eff - W_base|| / ||W_base|| over written rows
    dW = ff_out.effective_weight() - ff_out.W_base
    dose = (dW.norm() / (ff_out.W_base.norm() + 1e-8)).item()

    # logit-margin shift at the probe position (with vs without write)
    margin = 0.0
    nb = 6
    for _ in range(nb):
        Xp, Tp, _ = task.make_batch(args.batch, mapping, keys, device,
                                    include_query_pair=False)
        pos = (Tp != -100)
        ff_out.fast.zero_()
        base = model(Xp)
        ff_out.fast.copy_(fast)
        wr = model(Xp)
        idx = pos.nonzero(as_tuple=False)
        for (b, t) in idx:
            vt = int(Tp[b, t])
            margin += float(wr[b, t, vt] - base[b, t, vt])
    margin /= max(nb * args.batch, 1)

    acc = recall_acc(model, task, mapping, keys, include_query_pair=False,
                     device=device)

    ff_out.cfg.alpha = saved_alpha                    # restore
    ff_out.effective_weight = saved_eff
    for l in layers:
        l.reset_substrate()
    return acc, dose, margin


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    task = AssociativeRecall(n_keys=16, n_vals=16, pairs_per_seq=4, seed=args.seed)
    cfg = SubstrateConfig(group_size=4)
    chance = 1.0 / task.n_vals
    mlog = MetricsLog(Path(__file__).parent.parent / "metrics.jsonl",
                      run_name=f"oracle-v3-seed{args.seed}",
                      config={"seed": args.seed, "chance": chance,
                              "arm": "RULINGS3_oracle_v3_doseresponse"})

    torch.manual_seed(args.seed)
    model = TinyTransformer(task.vocab_size, d_model=args.d_model,
                            n_layers=args.layers, d_ff=args.d_ff, cfg=cfg,
                            use_substrate=True).to(device)
    print("--- pretrain (format acquisition) ---")
    pretrain(model, task, mapping_seed_range=10_000, steps=args.pretrain_steps,
             device=device)

    ff_out = model.blocks[-1].ff_out                  # LOCUS: last block ff_out
    assert isinstance(ff_out, SubstrateLinear)
    rt_mapping = task.random_mapping(task.held_keys, task.n_vals, 2)

    keyvec = clean_keys(model, ff_out, task, rt_mapping, task.held_keys,
                        device, args)
    M = build_M(model, ff_out, task, rt_mapping, task.held_keys, keyvec, device)

    print(f"\nOracle v3 dose-response (chance {chance:.3f}). "
          f"locus=last ff_out, clean key, value-embedding post.\n")
    print(f"{'budget':>7} {'alpha':>6} {'recall':>7} {'dose':>8} "
          f"{'logit_shift':>12}")
    grid = []
    for bf in args.budgets:                            # 'off' -> None
        bfv = None if bf == "off" else float(bf)
        for alpha in args.alphas:
            acc, dose, margin = measure_cell(model, ff_out, task, rt_mapping,
                                             task.held_keys, M, device, args,
                                             bfv, alpha)
            print(f"{str(bf):>7} {alpha:>6} {acc:>7.3f} {dose:>8.3f} "
                  f"{margin:>12.3f}")
            mlog.log(model="oracle_v3", phase="dose", budget=str(bf),
                     alpha=alpha, out_of_context_heldout=acc,
                     delivered_dose=dose, logit_shift=margin)
            grid.append({"budget": bf, "alpha": alpha, "recall": acc,
                         "dose": dose, "logit_shift": margin})

    # matched-magnitude linear-read cell (tanh->identity), budget off, alpha 1.0
    accL, doseL, marginL = measure_cell(model, ff_out, task, rt_mapping,
                                        task.held_keys, M, device, args,
                                        None, 1.0, linear=True)
    print(f"\nlinear-read (identity, budget off, alpha 1.0): recall {accL:.3f} "
          f"dose {doseL:.3f} logit_shift {marginL:.3f}")
    mlog.log(model="oracle_v3", phase="dose", budget="off", alpha=1.0,
             linear=True, out_of_context_heldout=accL, delivered_dose=doseL,
             logit_shift=marginL)
    grid.append({"budget": "off", "alpha": 1.0, "recall": accL, "dose": doseL,
                 "logit_shift": marginL, "linear": True})

    best = max(g["recall"] for g in grid)
    print("\n================ v3 VERDICT ================")
    print(f"best recall across full dose grid: {best:.3f}  (chance {chance:.3f})")
    if best > chance + 0.10:
        print("READ path WORKS. Additive bounded perturbation can retrieve. "
              "Earlier 0.000 were construction artifacts. -> retarget R2 to "
              "this locus (RULINGS_3 F4 success branch).")
    else:
        print("READ DEAD across all delivered doses. Additive perturbation of "
              "pretrained weights is not a viable memory locus. -> activate the "
              "dedicated associative-matrix redesign (RULINGS_3 F4 fail branch).")
    out = Path(__file__).parent.parent / "oracle_v3_results.json"
    out.write_text(json.dumps({"seed": args.seed, "chance": chance,
                               "grid": grid}, indent=2))
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
    p.add_argument("--oracle_batches", type=int, default=6)
    p.add_argument("--budgets", nargs="+", default=["0.25", "0.5", "1.0", "off"])
    p.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.4, 1.0])
    a = p.parse_args()
    if a.fast:
        a.d_model, a.layers, a.d_ff = 64, 2, 128
        a.pretrain_steps = 300
        a.oracle_batches = 3
    run(a)


if __name__ == "__main__":
    main()
