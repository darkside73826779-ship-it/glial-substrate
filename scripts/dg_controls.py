"""Contamination controls for the in-model 2x2 result (Rebecca's audit).

Before trusting recall=0.805, prove it is REAL key->value retrieval and not an
artifact of data exchange / leakage. For the two completion arms (raw, DG) at
the winning gamma, run:

  C1 NORMAL        : correct memory (reproduce the headline).
  C0 GAMMA=0       : no memory bias at all -> pure base model. Must be ~chance
                     (proves the memory, not the transformer, produces recall).
  C2 PERMUTED BIND : store each key's code bound to the WRONG value (roll by 1),
                     everything else identical. If recall stays high, the answer
                     is leaking independent of key->value binding = ARTIFACT.
                     If it collapses to ~chance, retrieval genuinely depends on
                     the correct binding = REAL.
  C3 SHUFFLED QUERY: read with a RANDOM held key's code instead of the true
                     query key's. Must collapse to ~chance if retrieval is
                     key-selective.
  C4 EMPTY MEMORY  : memory wiped (zero values) -> bias ~0. Must be ~chance.

Verdict: result is trustworthy iff C1 high AND {C0, C2, C3, C4} all ~chance.
Diagnostic; no learning; invariants unchanged.
Usage:  python scripts/dg_controls.py [--fast]
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
from run_experiment import pretrain
from oracle_probe_v3 import pair_absent_batch
from dg_inmodel import capture_key


@torch.no_grad()
def eval_completion(model, ff_out, task, mapping, keys, device, args, emb,
                    use_dg, dg, store_codes, store_vals, gamma, temp,
                    shuffle_query=False):
    """Completion-read recall with linear logit-bias injection. shuffle_query
    replaces each query code with a random OTHER key's stored code."""
    layers = model.substrate_layers()
    n = store_codes.shape[0]
    hits = total = 0
    for _ in range(args.eval_batches):
        X, T, _ = task.make_batch(args.batch, mapping, keys, device,
                                  include_query_pair=False)
        act = {}
        fmap = {l: torch.zeros(l.out_features, l.in_features, device=device)
                for l in layers}
        logits = model(X, fast_map=fmap, act_out=act)
        gq = act[ff_out][0]
        mask = (T != -100)
        for b in range(X.shape[0]):
            for t in range(X.shape[1]):
                if not mask[b, t]:
                    continue
                key = gq[b, t]
                qcode = (dg.encode(key.unsqueeze(0)).squeeze(0) if use_dg
                         else F.normalize(key, dim=-1))
                if shuffle_query:
                    qcode = store_codes[int(torch.randint(n, (1,)))]
                scores = store_codes @ qcode
                att = torch.softmax(scores / temp, dim=0)
                yhat = att @ store_vals
                bias = gamma * (yhat @ emb.t())
                pred = int((logits[b, t] + bias).argmax())
                hits += int(pred == int(T[b, t]))
                total += 1
    return hits / max(total, 1)


@torch.no_grad()
def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    task = AssociativeRecall(n_keys=16, n_vals=16, pairs_per_seq=4, seed=args.seed)
    cfg = SubstrateConfig(group_size=4)
    chance = 1.0 / task.n_vals
    mlog = MetricsLog(Path(__file__).parent.parent / "metrics.jsonl",
                      run_name=f"dg-controls-seed{args.seed}",
                      config={"seed": args.seed, "chance": chance,
                              "arm": "DG_contamination_controls"})

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

    store_keys = torch.stack([capture_key(model, ff_out, task, mapping, keys,
                                          k, device, args, 100 + i)
                              for i, k in enumerate(keys)])
    dg.adapt(store_keys)
    codes_dg = dg.encode(store_keys)
    codes_raw = F.normalize(store_keys, dim=-1)
    vals = torch.stack([emb[task.n_keys + mapping[k]] /
                        (emb[task.n_keys + mapping[k]].norm() + 1e-8)
                        for k in keys])
    vals_perm = vals[torch.roll(torch.arange(len(keys)), 1)]  # wrong binding
    vals_zero = torch.zeros_like(vals)

    print(f"\nraw overlap {mean_abs_cos(codes_raw):.3f}, DG overlap "
          f"{mean_abs_cos(codes_dg):.3f}, chance {chance:.3f}\n")
    print(f"{'arm':>16} {'C1 normal':>10} {'C0 g=0':>8} {'C2 permut':>10} "
          f"{'C3 shuffle':>11} {'C4 empty':>9}")
    results = {}
    for use_dg, sc in [(False, codes_raw), (True, codes_dg)]:
        name = "DG" if use_dg else "raw"
        c1 = eval_completion(model, ff_out, task, mapping, keys, device, args,
                             emb, use_dg, dg, sc, vals, args.gamma, args.temp)
        c0 = eval_completion(model, ff_out, task, mapping, keys, device, args,
                             emb, use_dg, dg, sc, vals, 0.0, args.temp)
        c2 = eval_completion(model, ff_out, task, mapping, keys, device, args,
                             emb, use_dg, dg, sc, vals_perm, args.gamma, args.temp)
        c3 = eval_completion(model, ff_out, task, mapping, keys, device, args,
                             emb, use_dg, dg, sc, vals, args.gamma, args.temp,
                             shuffle_query=True)
        c4 = eval_completion(model, ff_out, task, mapping, keys, device, args,
                             emb, use_dg, dg, sc, vals_zero, args.gamma, args.temp)
        print(f"{name+'+compl':>16} {c1:>10.3f} {c0:>8.3f} {c2:>10.3f} "
              f"{c3:>11.3f} {c4:>9.3f}")
        mlog.log(model="dg_controls", phase="control", arm=name, C1_normal=c1,
                 C0_gamma0=c0, C2_permuted=c2, C3_shuffled=c3, C4_empty=c4)
        results[name] = {"C1": c1, "C0": c0, "C2": c2, "C3": c3, "C4": c4}

    print("\n================ CONTAMINATION VERDICT ================")
    ok = True
    for name, r in results.items():
        controls_low = all(r[c] < chance + 0.15 for c in ("C0", "C2", "C3", "C4"))
        real = r["C1"] > 0.5 and controls_low
        ok = ok and (r["C1"] <= 0.5 or real)
        tag = ("REAL retrieval" if real else
               "SUSPECT — a control did not collapse" if r["C1"] > 0.5 else
               "below bar")
        print(f"  {name}+compl: C1={r['C1']:.3f}, controls "
              f"[{r['C0']:.3f},{r['C2']:.3f},{r['C3']:.3f},{r['C4']:.3f}] -> {tag}")
    print("\nOVERALL: results are TRUSTWORTHY (no leakage detected)."
          if ok else
          "\nOVERALL: CONTAMINATION SUSPECTED — a control stayed high; do not "
          "trust the 2x2 until explained.")
    out = Path(__file__).parent.parent / "dg_controls_results.json"
    out.write_text(json.dumps({"seed": args.seed, "chance": chance,
                               "results": results}, indent=2))
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
    p.add_argument("--gamma", type=float, default=10.0)
    a = p.parse_args()
    if a.fast:
        a.d_model, a.layers, a.d_ff = 64, 2, 128
        a.pretrain_steps = 300
        a.ctx_batches = 2
        a.eval_batches = 4
    run(a)


if __name__ == "__main__":
    main()
