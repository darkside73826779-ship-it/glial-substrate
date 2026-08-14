"""Side-by-side DG vs softmax under increasing memory load + query noise.

The regime that actually distinguishes them: store N items (N growing), query
out-of-context with noise, compare tuned-DG completion vs raw softmax completion.
Larger vocab (n_keys) so load can climb past the 16-item toy. Streams phase=
"load" to metrics.jsonl -> monitor 'Load test: DG vs softmax' card (both lines).

Slow, watchable: a short pause per point so the 2s dashboard poll catches each.
Diagnostic; no learning; invariants unchanged.
Usage:  python scripts/load_compare.py [--fast]
"""
from __future__ import annotations
import argparse, sys, time, json
from pathlib import Path
import torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from substrate import AssociativeRecall, SubstrateConfig, TinyTransformer
from substrate.dg import kwta
from substrate.metrics import MetricsLog
from run_experiment import pretrain
from dg_inmodel import capture_key

@torch.no_grad()
def recall(query_codes, store_codes, store_vals, emb, correct_tok, temp):
    att = torch.softmax((query_codes @ store_codes.t()) / temp, dim=1)
    yhat = att @ store_vals
    pred = (yhat @ emb.t()).argmax(1)
    return (pred == correct_tok).float().mean().item()

@torch.no_grad()
def run(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    task = AssociativeRecall(n_keys=args.n_keys, n_vals=args.n_keys,
                             pairs_per_seq=4, seed=args.seed)
    cfg = SubstrateConfig(group_size=4)
    chance = 1.0 / task.n_vals
    mlog = MetricsLog(Path(__file__).parent.parent / "metrics.jsonl",
                      run_name=f"load-compare-seed{args.seed}",
                      config={"seed": args.seed, "chance": chance,
                              "arm": "DG_vs_softmax_load", "n_keys": args.n_keys})
    torch.manual_seed(args.seed)
    model = TinyTransformer(task.vocab_size, d_model=args.d_model,
                            n_layers=args.layers, d_ff=args.d_ff, cfg=cfg,
                            use_substrate=True).to(dev)
    print("--- pretrain (once) ---")
    with torch.enable_grad():
        pretrain(model, task, mapping_seed_range=10_000,
                 steps=args.pretrain_steps, device=dev)
    ff = model.blocks[-1].ff_out
    mapping = task.random_mapping(task.held_keys, task.n_vals, 2)
    heldall = task.held_keys
    emb = model.tok.weight.detach()

    # fixed tuned DG config (set-and-forget) + a random frozen P
    g = torch.Generator().manual_seed(args.seed)
    P = (torch.randn(args.d_dg, ff.in_features, generator=g) /
         (ff.in_features ** 0.5)).to(dev)

    print(f"{'load':>5} {'noise':>6} {'DG':>7} {'softmax':>8} (chance {chance:.3f})")
    for load in args.loads:
        if load > len(heldall):
            continue
        keys = heldall[:load]
        skeys = torch.stack([capture_key(model, ff, task, mapping, keys, k, dev,
                             args, 100 + i) for i, k in enumerate(keys)])
        mu = skeys.mean(0)
        svals = torch.stack([emb[task.n_keys + mapping[k]] /
                             (emb[task.n_keys + mapping[k]].norm() + 1e-8)
                             for k in keys])
        correct = torch.tensor([task.n_keys + mapping[k] for k in keys], device=dev)
        raw_s = F.normalize(skeys, dim=-1)
        dg_s = F.normalize(kwta((skeys - mu) @ P.t(), args.k), dim=-1)
        for noise in args.noises:
            dgr, rawr = [], []
            for r in range(args.resamples):
                qk = torch.stack([capture_key(model, ff, task, mapping, keys, k,
                                  dev, args, 700 + r*53 + i)
                                  for i, k in enumerate(keys)])
                if noise > 0:
                    qk = qk + noise * torch.randn_like(qk)
                raw_q = F.normalize(qk, dim=-1)
                dg_q = F.normalize(kwta((qk - mu) @ P.t(), args.k), dim=-1)
                dgr.append(recall(dg_q, dg_s, svals, emb, correct, args.temp))
                rawr.append(recall(raw_q, raw_s, svals, emb, correct, args.raw_temp))
            dg = sum(dgr)/len(dgr); raw = sum(rawr)/len(rawr)
            print(f"{load:>5} {noise:>6.2f} {dg:>7.3f} {raw:>8.3f}")
            mlog.log(model="load_compare", phase="load", load=load, noise=noise,
                     dg_recall=dg, softmax_recall=raw, out_of_context_heldout=dg)
            time.sleep(args.pause)
    out = Path(__file__).parent.parent / "load_compare_results.json"
    out.write_text(json.dumps({"seed": args.seed, "n_keys": args.n_keys}, indent=2))
    print("done")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fast", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_keys", type=int, default=128)  # held pool = 64
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=256)
    p.add_argument("--pretrain_steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--ctx_batches", type=int, default=4)
    p.add_argument("--resamples", type=int, default=4)
    p.add_argument("--d_dg", type=int, default=4096)
    p.add_argument("--k", type=int, default=24)
    p.add_argument("--temp", type=float, default=0.03)
    p.add_argument("--raw_temp", type=float, default=0.05)
    p.add_argument("--loads", type=int, nargs="+", default=[8,16,24,32,48,64])
    p.add_argument("--noises", type=float, nargs="+", default=[0.0, 0.1, 0.25])
    p.add_argument("--pause", type=float, default=0.6)
    a = p.parse_args()
    if a.fast:
        a.n_keys=64; a.d_model,a.layers,a.d_ff=64,2,128; a.pretrain_steps=300
        a.loads=[8,16,24,32]; a.noises=[0.0,0.25]; a.resamples=2
    run(a)

if __name__ == "__main__":
    main()
