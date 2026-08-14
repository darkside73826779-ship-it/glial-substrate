"""Hybrid encodings: merge softmax's noise tolerance with DG's high-load
capacity. Same load x noise grid, all variants side by side.

Variants (all read via softmax completion + linear logit bias):
  raw        dense normalized key (softmax parent: noise-robust, lower capacity)
  dg_hard    expand + hard kWTA k  (DG parent: high capacity, noise-brittle)
  dg_dense   expand, NO sparsify   (capacity from expansion, no hard cutoff)
  dg_soft    expand + SOFT sparsify (graded gate around k-th value; tau)
  blend      0.5*dg_hard + 0.5*raw retrieved values (ensemble of parents)

Streams phase="load" (dg_recall=best hybrid, softmax_recall=raw) to the monitor
load card; full table printed. Diagnostic; no learning; invariants unchanged.
Usage:  python scripts/hybrid_compare.py [--fast]
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

def soft_sparsify(z, k, tau):
    """Graded sparsify: relu then sigmoid gate around the k-th largest value —
    near-threshold units fade smoothly instead of hard on/off (noise robust)."""
    a = F.relu(z)
    if k >= a.shape[-1]:
        return a
    thresh = a.topk(k, dim=-1).values[..., -1:]
    return a * torch.sigmoid((a - thresh) / max(tau, 1e-6))

@torch.no_grad()
def retrieve(qc, sc, svals, temp):
    att = torch.softmax((qc @ sc.t()) / temp, dim=1)
    return att @ svals            # (P, d_model) retrieved value vectors

@torch.no_grad()
def acc(yhat, emb, correct):
    return ((yhat @ emb.t()).argmax(1) == correct).float().mean().item()

@torch.no_grad()
def run(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    task = AssociativeRecall(n_keys=args.n_keys, n_vals=args.n_keys,
                             pairs_per_seq=4, seed=args.seed)
    cfg = SubstrateConfig(group_size=4); chance = 1.0/task.n_vals
    mlog = MetricsLog(Path(__file__).parent.parent / "metrics.jsonl",
                      run_name=f"hybrid-seed{args.seed}",
                      config={"seed": args.seed, "chance": chance,
                              "arm": "hybrid_dg_softmax", "n_keys": args.n_keys})
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
    heldall = task.held_keys; emb = model.tok.weight.detach()
    g = torch.Generator().manual_seed(args.seed)
    P = (torch.randn(args.d_dg, ff.in_features, generator=g) /
         (ff.in_features**0.5)).to(dev)
    variants = ["raw","dg_hard","dg_dense","dg_soft","blend"]
    print(f"{'load':>4} {'noise':>5} " + " ".join(f"{v:>8}" for v in variants))
    for load in args.loads:
        if load > len(heldall): continue
        keys = heldall[:load]
        sk = torch.stack([capture_key(model,ff,task,mapping,keys,k,dev,args,100+i)
                          for i,k in enumerate(keys)])
        mu = sk.mean(0)
        sv = torch.stack([emb[task.n_keys+mapping[k]]/
                          (emb[task.n_keys+mapping[k]].norm()+1e-8) for k in keys])
        cor = torch.tensor([task.n_keys+mapping[k] for k in keys], device=dev)
        S = {"raw": F.normalize(sk,dim=-1),
             "dg_hard": F.normalize(kwta((sk-mu)@P.t(),args.k),dim=-1),
             "dg_dense": F.normalize(F.relu((sk-mu)@P.t()),dim=-1),
             "dg_soft": F.normalize(soft_sparsify((sk-mu)@P.t(),args.k,args.tau),dim=-1)}
        for noise in args.noises:
            res = {v: [] for v in variants}
            for r in range(args.resamples):
                qk = torch.stack([capture_key(model,ff,task,mapping,keys,k,dev,
                                  args,700+r*53+i) for i,k in enumerate(keys)])
                if noise>0: qk = qk + noise*torch.randn_like(qk)
                Q = {"raw": F.normalize(qk,dim=-1),
                     "dg_hard": F.normalize(kwta((qk-mu)@P.t(),args.k),dim=-1),
                     "dg_dense": F.normalize(F.relu((qk-mu)@P.t()),dim=-1),
                     "dg_soft": F.normalize(soft_sparsify((qk-mu)@P.t(),args.k,args.tau),dim=-1)}
                y = {v: retrieve(Q[v if v!='blend' else 'raw'],
                                 S[v if v!='blend' else 'raw'], sv,
                                 args.temp if v!='raw' else args.raw_temp)
                     for v in ["raw","dg_hard","dg_dense","dg_soft"]}
                y["blend"] = 0.5*y["dg_hard"] + 0.5*y["raw"]
                for v in variants: res[v].append(acc(y[v],emb,cor))
            row = {v: sum(res[v])/len(res[v]) for v in variants}
            print(f"{load:>4} {noise:>5.2f} " + " ".join(f"{row[v]:>8.3f}" for v in variants))
            best_h = max(row["dg_hard"],row["dg_dense"],row["dg_soft"],row["blend"])
            mlog.log(model="hybrid", phase="load", load=load, noise=noise,
                     dg_recall=best_h, softmax_recall=row["raw"],
                     out_of_context_heldout=best_h, **{f"v_{v}":row[v] for v in variants})
            time.sleep(args.pause)
    print("done")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fast",action="store_true"); p.add_argument("--seed",type=int,default=0)
    p.add_argument("--n_keys",type=int,default=128); p.add_argument("--d_model",type=int,default=128)
    p.add_argument("--layers",type=int,default=4); p.add_argument("--d_ff",type=int,default=256)
    p.add_argument("--pretrain_steps",type=int,default=3000); p.add_argument("--batch",type=int,default=16)
    p.add_argument("--ctx_batches",type=int,default=4); p.add_argument("--resamples",type=int,default=4)
    p.add_argument("--d_dg",type=int,default=4096); p.add_argument("--k",type=int,default=24)
    p.add_argument("--temp",type=float,default=0.03); p.add_argument("--raw_temp",type=float,default=0.05)
    p.add_argument("--tau",type=float,default=0.15)
    p.add_argument("--loads",type=int,nargs="+",default=[16,32,48,64])
    p.add_argument("--noises",type=float,nargs="+",default=[0.0,0.1,0.25,0.4])
    p.add_argument("--pause",type=float,default=0.6)
    a=p.parse_args()
    if a.fast:
        a.n_keys=64;a.d_model,a.layers,a.d_ff=64,2,128;a.pretrain_steps=300
        a.loads=[16,32];a.noises=[0.0,0.25];a.resamples=2
    run(a)

if __name__=="__main__":
    main()
