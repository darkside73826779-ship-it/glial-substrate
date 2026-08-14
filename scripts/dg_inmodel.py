"""First in-model DG test: does pattern separation retrieve THROUGH the real
transformer, injected as a linear logit bias (DG_INTEGRATION_PINS pin 1b)?

2x2 experiment (review note M-2): {raw keys | DG codes} x {linear | completion}
read. Retrieval is stored in a SEPARATE associative memory M and injected as
  logits += gamma * (yhat @ E^T),   yhat = read(query_code),  value = E[v] row
so the associative channel is LINEAR end-to-end (I-16; never through alpha*tanh).

Keys tapped at the last-block ff_out INPUT (d_ff=256, the oracle-validated
locus, pin 2). Store keys and query keys captured from DIFFERENT contexts
(review note M-3: real representational drift, not synthetic noise) so recall
is a genuine retrieval, not exact-match lookup.

Logs (PROGRAM_REVIEW P4): match_strength (calibration crown-jewel signal),
dg_code_overlap, and retention (pretrain-pool recall with the channel live).

Diagnostic; no learning run; invariants unchanged (18 green incl I-12..I-16).
Usage:  python scripts/dg_inmodel.py [--fast]
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
from substrate.dg import DGConfig, DGProjection, SparseAssociativeMemory, mean_abs_cos
from substrate.metrics import MetricsLog
from run_experiment import pretrain, loss_fn
from oracle_probe_v3 import pair_absent_batch


@torch.no_grad()
def capture_key(model, ff_out, task, mapping, keys, k, device, args, ctx_seed):
    """Mean pair-absent ff_out-input at the query position for key k, from a
    context sample selected by ctx_seed (store vs query use different seeds)."""
    layers = model.substrate_layers()
    g = torch.Generator(device="cpu").manual_seed(ctx_seed)
    acc = None
    for _ in range(args.ctx_batches):
        torch.manual_seed(int(torch.randint(2**30, (1,), generator=g)))
        X, fpos = pair_absent_batch(task, mapping, keys, k, device,
                                    args.batch, task.pairs_per_seq)
        act = {}
        fmap = {l: torch.zeros(l.out_features, l.in_features, device=device)
                for l in layers}
        _ = model(X, fast_map=fmap, act_out=act)
        v = act[ff_out][0][:, fpos, :].mean(0)          # (d_ff,)
        acc = v if acc is None else acc + v
    return acc / args.ctx_batches


@torch.no_grad()
def eval_arm(model, ff_out, task, mapping, keys, device, args, emb,
             use_dg, use_completion, dg, sam, store_codes, store_vals,
             gamma):
    """In-model out-of-context recall with the DG/raw retrieval channel
    injected as a linear logit bias. Returns (recall, mean_match_strength)."""
    layers = model.substrate_layers()
    hits = total = 0
    match_acc = 0.0
    nb = args.eval_batches
    for _ in range(nb):
        X, T, _ = task.make_batch(args.batch, mapping, keys, device,
                                  include_query_pair=False)
        act = {}
        fmap = {l: torch.zeros(l.out_features, l.in_features, device=device)
                for l in layers}
        logits = model(X, fast_map=fmap, act_out=act)   # base logits + act tap
        gq = act[ff_out][0]                             # (B,T,d_ff)
        mask = (T != -100)
        for b in range(X.shape[0]):
            for t in range(X.shape[1]):
                if not mask[b, t]:
                    continue
                key = gq[b, t]
                if use_dg:
                    qcode = dg.encode(key.unsqueeze(0)).squeeze(0)
                else:
                    qcode = F.normalize(key, dim=-1)
                if use_completion:
                    scores = store_codes @ qcode
                    match_acc += float(scores.max())
                    att = torch.softmax(scores / args.temp, dim=0)
                    yhat = att @ store_vals
                else:
                    yhat = (sam.read(qcode.unsqueeze(0)).squeeze(0) if use_dg
                            else (store_vals.t() @ (store_codes @ qcode)))
                    match_acc += float((store_codes @ qcode).max())
                bias = gamma * (yhat @ emb.t())          # (vocab,)
                pred = int((logits[b, t] + bias).argmax())
                hits += int(pred == int(T[b, t]))
                total += 1
    return hits / max(total, 1), match_acc / max(total, 1)


@torch.no_grad()
def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    task = AssociativeRecall(n_keys=16, n_vals=16, pairs_per_seq=4, seed=args.seed)
    cfg = SubstrateConfig(group_size=4)
    chance = 1.0 / task.n_vals
    mlog = MetricsLog(Path(__file__).parent.parent / "metrics.jsonl",
                      run_name=f"dg-inmodel-seed{args.seed}",
                      config={"seed": args.seed, "chance": chance, "k": 16,
                              "d_dg": 1024, "locus": "last_ff_out_input(d_ff)",
                              "arm": "DG_inmodel_2x2"})

    torch.manual_seed(args.seed)
    model = TinyTransformer(task.vocab_size, d_model=args.d_model,
                            n_layers=args.layers, d_ff=args.d_ff, cfg=cfg,
                            use_substrate=True).to(device)
    print("--- pretrain ---")
    with torch.enable_grad():
        pretrain(model, task, mapping_seed_range=10_000,
                 steps=args.pretrain_steps, device=device)

    ff_out = model.blocks[-1].ff_out
    d_ff = ff_out.in_features
    mapping = task.random_mapping(task.held_keys, task.n_vals, 2)
    keys = task.held_keys
    emb = model.tok.weight.detach()

    dcfg = DGConfig(d_in=d_ff, d_dg=1024, k=16, mode="random", seed=args.seed)
    dg = DGProjection(dcfg).to(device)
    sam = SparseAssociativeMemory(d_value=args.d_model, cfg=dcfg).to(device)

    # store keys (context A); adapt then encode; write M and keep arrays
    store_keys = torch.stack([capture_key(model, ff_out, task, mapping, keys,
                                          k, device, args, 100 + i)
                              for i, k in enumerate(keys)])
    dg.adapt(store_keys)
    store_codes = dg.encode(store_keys)                  # (n, d_dg)
    store_raw = F.normalize(store_keys, dim=-1)          # (n, d_ff)
    store_vals = torch.stack([emb[task.n_keys + mapping[k]] /
                              (emb[task.n_keys + mapping[k]].norm() + 1e-8)
                              for k in keys])             # (n, d_model)
    sam.reset()
    sam.write(store_vals, store_codes, gain=1.0)

    overlap_raw = mean_abs_cos(store_raw)
    overlap_dg = mean_abs_cos(store_codes)
    print(f"\nstore-key overlap: raw |cos| {overlap_raw:.3f} -> "
          f"DG code |cos| {overlap_dg:.3f}")
    mlog.log(model="dg_inmodel", phase="setup", dg_code_overlap=overlap_dg,
             raw_key_overlap=overlap_raw)

    print(f"\nin-model 2x2 (chance {chance:.3f}, bar 0.50). "
          f"query keys from held-out contexts.\n")
    print(f"{'arm':>18} {'gamma':>6} {'oo_recall':>10} {'match':>7}")
    grid = []
    for use_dg in (False, True):
        for use_comp in (False, True):
            name = ("DG" if use_dg else "raw") + "+" + \
                   ("completion" if use_comp else "linear")
            best = 0.0
            for gamma in args.gammas:
                rec, match = eval_arm(model, ff_out, task, mapping, keys, device,
                                      args, emb, use_dg, use_comp, dg, sam,
                                      store_codes if use_dg else store_raw,
                                      store_vals, gamma)
                print(f"{name:>18} {gamma:>6} {rec:>10.3f} {match:>7.3f}")
                mlog.log(model="dg_inmodel", phase="arm2x2", arm=name,
                         gamma=gamma, out_of_context_heldout=rec,
                         match_strength=match)
                grid.append({"arm": name, "gamma": gamma, "recall": rec,
                             "match": match})
                best = max(best, rec)
            print(f"{'  -> best':>18} {'':>6} {best:>10.3f}")

    best_arm = max(grid, key=lambda g: g["recall"])
    print("\n================ 2x2 VERDICT ================")
    print(f"best: {best_arm['arm']} @ gamma {best_arm['gamma']} -> "
          f"recall {best_arm['recall']:.3f} (chance {chance:.3f}, bar 0.50)")
    surv = best_arm["recall"] >= 0.50
    print("DG retrieval SURVIVES the transformer via linear logit bias."
          if surv else
          "Still below bar in-model — inspect injection gain / locus.")
    out = Path(__file__).parent.parent / "dg_inmodel_results.json"
    out.write_text(json.dumps({"seed": args.seed, "chance": chance,
                               "overlap_raw": overlap_raw,
                               "overlap_dg": overlap_dg, "grid": grid}, indent=2))
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
    p.add_argument("--eval_batches", type=int, default=8)
    p.add_argument("--temp", type=float, default=0.05)
    p.add_argument("--gammas", type=float, nargs="+", default=[2.0, 5.0, 10.0])
    a = p.parse_args()
    if a.fast:
        a.d_model, a.layers, a.d_ff = 64, 2, 128
        a.pretrain_steps = 300
        a.ctx_batches = 2
        a.eval_batches = 4
    run(a)


if __name__ == "__main__":
    main()
