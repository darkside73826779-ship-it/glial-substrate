"""DG viability sweep (Rebecca): many runs varying DG's controls, streaming the
accuracy curve so we can watch it rise/fall and judge the module honestly.

Efficiency: the transformer forward is INDEPENDENT of DG config, so we pretrain
once and cache the eval set (base logits + query keys + targets at answer
positions). Every DG config is then pure tensor post-processing on the cache —
thousands of configs stream in seconds. The raw+completion baseline (DG off) is
recomputed at each temperature as the reference line.

Swept controls: d_dg (expansion), k (sparsity), temp (completion softmax — the
confound to resolve fairly), gamma (injection gain). Logged per config:
recall, DG code overlap, match strength. Streams phase="sweep" to metrics.jsonl
(monitor 'DG sweep' card). Verdict tracks best DG vs best raw continuously.

Diagnostic; no learning; invariants unchanged.
Usage:  python scripts/dg_sweep.py [--fast]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from substrate import AssociativeRecall, SubstrateConfig, TinyTransformer
from substrate.dg import DGConfig, DGProjection, kwta, mean_abs_cos
from substrate.metrics import MetricsLog
from run_experiment import pretrain
from oracle_probe_v3 import pair_absent_batch
from dg_inmodel import capture_key


@torch.no_grad()
def build_eval_cache(model, ff_out, task, mapping, keys, device, args):
    """Collect (base_logits, query_key, target) at answer positions across many
    out-of-context probe batches. Config-independent -> computed once."""
    layers = model.substrate_layers()
    L, Q, Y = [], [], []
    for _ in range(args.cache_batches):
        X, T, _ = task.make_batch(args.batch, mapping, keys, device,
                                  include_query_pair=False)
        act = {}
        fmap = {l: torch.zeros(l.out_features, l.in_features, device=device)
                for l in layers}
        logits = model(X, fast_map=fmap, act_out=act)
        gq = act[ff_out][0]
        idx = (T != -100).nonzero(as_tuple=False)
        for (b, t) in idx:
            L.append(logits[b, t]); Q.append(gq[b, t]); Y.append(int(T[b, t]))
    return torch.stack(L), torch.stack(Q), torch.tensor(Y, device=device)


@torch.no_grad()
def recall_of(base_logits, query_codes, store_codes, store_vals, emb, targets,
              temp, gamma):
    scores = query_codes @ store_codes.t()               # (P, n)
    att = torch.softmax(scores / temp, dim=1)
    yhat = att @ store_vals                               # (P, d_model)
    bias = gamma * (yhat @ emb.t())                       # (P, vocab)
    pred = (base_logits + bias).argmax(dim=1)
    match = scores.max(dim=1).values.mean().item()
    return (pred == targets).float().mean().item(), match


@torch.no_grad()
def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    task = AssociativeRecall(n_keys=16, n_vals=16, pairs_per_seq=4, seed=args.seed)
    cfg = SubstrateConfig(group_size=4)
    chance = 1.0 / task.n_vals
    mlog = MetricsLog(Path(__file__).parent.parent / "metrics.jsonl",
                      run_name=f"dg-sweep-seed{args.seed}",
                      config={"seed": args.seed, "chance": chance,
                              "arm": "DG_viability_sweep"})

    torch.manual_seed(args.seed)
    model = TinyTransformer(task.vocab_size, d_model=args.d_model,
                            n_layers=args.layers, d_ff=args.d_ff, cfg=cfg,
                            use_substrate=True).to(device)
    print("--- pretrain (once) ---")
    with torch.enable_grad():
        pretrain(model, task, mapping_seed_range=10_000,
                 steps=args.pretrain_steps, device=device)

    ff_out = model.blocks[-1].ff_out
    d_ff = ff_out.in_features
    mapping = task.random_mapping(task.held_keys, task.n_vals, 2)
    keys = task.held_keys
    emb = model.tok.weight.detach()

    store_keys = torch.stack([capture_key(model, ff_out, task, mapping, keys,
                                          k, device, args, 100 + i)
                              for i, k in enumerate(keys)])
    mu = store_keys.mean(0)
    store_vals = torch.stack([emb[task.n_keys + mapping[k]] /
                              (emb[task.n_keys + mapping[k]].norm() + 1e-8)
                              for k in keys])
    base_logits, query_keys, targets = build_eval_cache(
        model, ff_out, task, mapping, keys, device, args)
    print(f"eval cache: {base_logits.shape[0]} answer positions\n")

    # raw baseline reference (DG off): best over temp x gamma
    raw_sc = F.normalize(store_keys, dim=-1)
    raw_qc = F.normalize(query_keys, dim=-1)
    raw_best = 0.0
    for temp, gamma in itertools.product(args.temps, args.gammas):
        r, _ = recall_of(base_logits, raw_qc, raw_sc, store_vals, emb,
                         targets, temp, gamma)
        raw_best = max(raw_best, r)
    print(f"raw+completion best (reference): {raw_best:.3f}\n")
    mlog.log(model="dg_sweep", phase="baseline", raw_best=raw_best)

    print(f"{'#':>4} {'d_dg':>5} {'k':>4} {'temp':>6} {'gamma':>6} "
          f"{'recall':>7} {'overlap':>8} {'bestDG':>7}")
    best = {"recall": 0.0}
    allcells = []
    step = 0
    for d_dg in args.dgs:
        g = torch.Generator().manual_seed(args.seed)
        P = (torch.randn(d_dg, d_ff, generator=g) / (d_ff ** 0.5)).to(device)
        zs = (store_keys - mu) @ P.t()
        zq = (query_keys - mu) @ P.t()
        for k in args.ks:
            if k > d_dg:
                continue
            sc = F.normalize(kwta(zs, k), dim=-1)
            qc = F.normalize(kwta(zq, k), dim=-1)
            overlap = mean_abs_cos(sc)
            for temp, gamma in itertools.product(args.temps, args.gammas):
                rec, match = recall_of(base_logits, qc, sc, store_vals, emb,
                                       targets, temp, gamma)
                step += 1
                if rec > best["recall"]:
                    best = {"recall": rec, "d_dg": d_dg, "k": k,
                            "temp": temp, "gamma": gamma}
                mlog.log(model="dg_sweep", phase="sweep", step=step, d_dg=d_dg,
                         k=k, temp=temp, gamma=gamma, out_of_context_heldout=rec,
                         dg_code_overlap=overlap, match_strength=match,
                         best_dg=best["recall"], raw_best=raw_best)
                allcells.append({"d_dg": d_dg, "k": k, "temp": temp,
                                 "gamma": gamma, "recall": rec,
                                 "overlap": overlap})
                if step % args.print_every == 0:
                    print(f"{step:>4} {d_dg:>5} {k:>4} {temp:>6.3f} {gamma:>6.1f} "
                          f"{rec:>7.3f} {overlap:>8.3f} {best['recall']:>7.3f}")

    print("\n================ DG VIABILITY VERDICT ================")
    print(f"configs swept: {step}")
    print(f"raw+completion best: {raw_best:.3f}")
    print(f"DG+completion best:  {best['recall']:.3f}  @ {best}")
    delta = best["recall"] - raw_best
    if delta > 0.03:
        print(f"DG BEATS raw by {delta:+.3f} at its best point -> the module "
              f"earns its place; separation adds value once temp-calibrated.")
    elif delta > -0.03:
        print(f"DG MATCHES raw ({delta:+.3f}) -> separation is neutral here; "
              f"completion carries retrieval. Keep DG only if it helps capacity/"
              f"retention under load (next study), else it is optional.")
    else:
        print(f"DG UNDERPERFORMS raw by {delta:+.3f} even tuned -> at toy scale "
              f"the module is not viable for recall; completion suffices.")
    out = Path(__file__).parent.parent / "dg_sweep_results.json"
    out.write_text(json.dumps({"seed": args.seed, "chance": chance,
                               "raw_best": raw_best, "dg_best": best,
                               "configs": step, "grid": allcells}, indent=2))
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
    p.add_argument("--cache_batches", type=int, default=16)
    p.add_argument("--print_every", type=int, default=25)
    p.add_argument("--dgs", type=int, nargs="+", default=[512, 1024, 2048])
    p.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128])
    p.add_argument("--temps", type=float, nargs="+",
                   default=[0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.5])
    p.add_argument("--gammas", type=float, nargs="+", default=[5.0, 10.0, 20.0])
    a = p.parse_args()
    if a.fast:
        a.d_model, a.layers, a.d_ff = 64, 2, 128
        a.pretrain_steps = 300
        a.cache_batches = 6
        a.dgs = [512, 1024]
        a.ks = [8, 16, 32]
    run(a)


if __name__ == "__main__":
    main()
