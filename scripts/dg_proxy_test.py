"""DG-proxy de-risk harness (pre-staged for the Rebecca+Claude dentate-gyrus design).

Answers the one question that de-risks the whole approach BEFORE the in-model
integration: does pattern separation (expand + sparsify, k-WTA) + a clean
associative read (Hopfield softmax, no alpha*tanh weight path) retrieve where
whitening + the additive read failed (step 1: offline 1.000 -> in-model 0.000)?

Pipeline (fixed-random DG default; swap in Claude's spec at PLUG POINTS):
  clean key g (d_ff, pair-absent forward)
   -> EXPAND:   h = relu(R g),  R random (E x d_ff), fixed              [PLUG: R]
   -> SPARSIFY: k-WTA, keep top-k, L2-normalize -> sparse code (E)      [PLUG: k]
  store  M = {(dg(key_k), value_emb_k)}
  READ (Hopfield):  s_j = code_q . code_j ; w = softmax(beta s) ;
                    retrieved = Σ_j w_j value_emb_j ; logits = emb[vals] @ retrieved
Non-trivial retrieval: store keys and query keys captured from DIFFERENT context
batches, so this is not exact-match lookup — a mild robustness/generalization
signal (full learned-codec generalization gate comes with Claude's learned P).

Controls in the same run:
  - RAW read  (no DG): softmax read on raw clean keys — isolates the DG effect.
  - DG read   (this):   expand+sparsify then softmax read.
Sweeps expansion E, sparsity k, temperature beta.

Diagnostic only; no core change, no invariant touched, no learning run.
Usage:  python scripts/dg_proxy_test.py [--fast]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from substrate import AssociativeRecall, SubstrateConfig, TinyTransformer
from substrate.metrics import MetricsLog
from run_experiment import pretrain
from oracle_probe_v3 import pair_absent_batch


@torch.no_grad()
def capture_keys(model, ff_out, task, mapping, keys, device, args, seed_off):
    """Mean pair-absent query-position ff_out input per key, from a context
    sample distinct per seed_off (store vs query use different samples)."""
    layers = model.substrate_layers()
    g = torch.Generator(device="cpu").manual_seed(1234 + seed_off)
    out = {}
    for k in keys:
        acc = None
        for _ in range(args.ctx_batches):
            torch.manual_seed(int(torch.randint(2**30, (1,), generator=g)))
            X, fpos = pair_absent_batch(task, mapping, keys, k, device,
                                        args.batch, task.pairs_per_seq)
            act = {}
            fmap = {l: torch.zeros(l.out_features, l.in_features, device=device)
                    for l in layers}
            _ = model(X, fast_map=fmap, act_out=act)
            v = act[ff_out][0][:, fpos, :].mean(0)
            acc = v if acc is None else acc + v
        out[k] = acc / args.ctx_batches
    return out


def dg_code(g, R, k):
    """Expand + sparsify (k-WTA). g:(...,d_ff) -> sparse (...,E)."""
    h = torch.relu(g @ R.t())                       # (..., E)   [PLUG POINT: R]
    if k < h.shape[-1]:
        thresh = h.topk(k, dim=-1).values[..., -1:]
        h = torch.where(h >= thresh, h, torch.zeros_like(h))
    return h / (h.norm(dim=-1, keepdim=True) + 1e-8)


@torch.no_grad()
def hopfield_recall(store_codes, val_embs, query_codes, Vtoks, emb, val_range,
                    beta):
    """Softmax associative read -> value logits -> argmax. No alpha*tanh path."""
    n = store_codes.shape[0]
    cand = emb[val_range]
    hits = 0
    for i in range(n):
        s = store_codes @ query_codes[i]            # (n,)
        w = torch.softmax(beta * s, dim=0)
        retrieved = (w.unsqueeze(1) * val_embs).sum(0)   # (d_model,)
        pred = val_range[int((cand @ retrieved).argmax())]
        hits += int(pred == Vtoks[i])
    return hits / max(n, 1)


@torch.no_grad()
def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    task = AssociativeRecall(n_keys=16, n_vals=16, pairs_per_seq=4, seed=args.seed)
    cfg = SubstrateConfig(group_size=4)
    chance = 1.0 / task.n_vals
    mlog = MetricsLog(Path(__file__).parent.parent / "metrics.jsonl",
                      run_name=f"dg-proxy-seed{args.seed}",
                      config={"seed": args.seed, "chance": chance,
                              "arm": "DG_proxy_derisk"})

    torch.manual_seed(args.seed)
    model = TinyTransformer(task.vocab_size, d_model=args.d_model,
                            n_layers=args.layers, d_ff=args.d_ff, cfg=cfg,
                            use_substrate=True).to(device)
    print("--- pretrain ---")
    with torch.enable_grad():
        pretrain(model, task, mapping_seed_range=10_000,
                 steps=args.pretrain_steps, device=device)

    ff_out = model.blocks[-1].ff_out
    mapping = task.random_mapping(task.held_keys, task.n_vals, 2)
    keys = task.held_keys
    emb = model.tok.weight.detach()
    val_range = list(range(task.n_keys, task.n_keys + task.n_vals))
    Vtoks = [task.n_keys + mapping[k] for k in keys]
    val_embs = torch.stack([emb[t] / (emb[t].norm() + 1e-8) for t in Vtoks])

    # store-set and query-set from DIFFERENT contexts (non-trivial retrieval)
    store_raw = capture_keys(model, ff_out, task, mapping, keys, device, args, 0)
    query_raw = capture_keys(model, ff_out, task, mapping, keys, device, args, 7)
    Ks = torch.stack([store_raw[k] / (store_raw[k].norm() + 1e-8) for k in keys])
    Kq = torch.stack([query_raw[k] / (query_raw[k].norm() + 1e-8) for k in keys])

    # baseline collinearity of the raw keys (context-varied)
    G = Ks @ Kq.t()
    diag = G.diag().mean().item()
    off = G[~torch.eye(len(keys), dtype=torch.bool, device=device)].abs().mean().item()
    print(f"\nraw key store/query: same-key cos {diag:.3f}, cross-key |cos| "
          f"{off:.3f}  (chance {chance:.3f})")

    # RAW control (no DG): softmax read on raw keys
    raw_best = 0.0
    for beta in args.betas:
        r = hopfield_recall(Ks, val_embs, Kq, Vtoks, emb, val_range, beta)
        raw_best = max(raw_best, r)
    print(f"[control] RAW softmax read best over beta: {raw_best:.3f}")

    print(f"\nDG proxy (expand+kWTA) softmax read. chance {chance:.3f}\n")
    print(f"{'E':>6} {'k':>5} {'beta':>6} {'recall':>8}")
    grid = []
    for E in args.expansions:
        Rmat = torch.randn(E, args.d_ff, device=device) / (args.d_ff ** 0.5)
        for k in args.ks:
            if k > E:
                continue
            sc = dg_code(Ks, Rmat, k)
            qc = dg_code(Kq, Rmat, k)
            for beta in args.betas:
                rec = hopfield_recall(sc, val_embs, qc, Vtoks, emb, val_range, beta)
                print(f"{E:>6} {k:>5} {beta:>6} {rec:>8.3f}")
                mlog.log(model="dg_proxy", phase="dg", expansion=E, k=k,
                         beta=beta, out_of_context_heldout=rec)
                grid.append({"E": E, "k": k, "beta": beta, "recall": rec})

    best = max(g["recall"] for g in grid)
    print("\n================ DG DE-RISK VERDICT ================")
    print(f"raw-read best {raw_best:.3f}  |  DG-read best {best:.3f}  "
          f"(chance {chance:.3f})")
    if best > 0.5 and best > raw_best + 0.1:
        print("Pattern separation + clean read RETRIEVES where additive/tanh "
              "failed. DG proxy is the right instrument. -> integrate in-model "
              "(inject retrieved value to residual) + learned-P generalization gate.")
    elif best > raw_best + 0.1:
        print("DG helps over raw but below 0.5 — sweep sparsity/expansion or "
              "pair with learned projection.")
    else:
        print("DG (fixed-random) does not beat raw read here — likely need a "
              "LEARNED separation (Claude's learned P), or richer context-robust "
              "keys. Data logged for the design.")
    out = Path(__file__).parent.parent / "dg_proxy_results.json"
    out.write_text(json.dumps({"seed": args.seed, "chance": chance,
                               "raw_best": raw_best, "same_key_cos": diag,
                               "cross_key_cos": off, "grid": grid}, indent=2))
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
    p.add_argument("--ctx_batches", type=int, default=6)
    p.add_argument("--expansions", type=int, nargs="+", default=[512, 1024, 2048])
    p.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32, 64])
    p.add_argument("--betas", type=float, nargs="+", default=[4.0, 8.0, 16.0])
    a = p.parse_args()
    if a.fast:
        a.d_model, a.layers, a.d_ff = 64, 2, 128
        a.pretrain_steps = 300
        a.ctx_batches = 3
        a.expansions = [512, 1024]
        a.ks = [8, 32]
    run(a)


if __name__ == "__main__":
    main()
