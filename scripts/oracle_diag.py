"""Oracle diagnostic (RULINGS_3 V2) — WHY is the read non-selective?

Pure measurement + offline associative-read math. NO learning run, NO invariant
or bound changes. Isolates the outer-product read from the transformer to split
the sub-fork:
  - locus-fundamental (the storage location can't retrieve) vs
  - key-orthogonality (the location could, but the pretrained keys overlap).

Three probes on the last block's ff_out clean keys (pair-absent forwards):
  A. KEY OVERLAP: pairwise cosine similarity of the clean keys. High off-diagonal
     = keys are not separable -> non-selective read explained by key correlation.
  B. OFFLINE READ (real keys): build M = Σ outer(emb[value_k], key_k); for each
     query key k', argmax over value tokens of emb @ (M @ key_k'). This is the
     associative read in ISOLATION (no tanh, no budget, no residual/ln/head
     competition). If this is high but in-model v3 recall is 0.000, the storage
     math is fine and the transformer downstream destroys it (locus/integration).
     If this is at chance, the real keys themselves cannot be separated.
  C. CEILING (orthonormal keys): same read with an orthonormal key set (store and
     query orthonormal). Expect ~1.0 — proves the outer-product mechanism works
     when keys ARE separable. The gap B->C is the cost of key non-orthogonality.

Verdict logic:
  C high, B low, A high  -> KEY-ORTHOGONALITY (F4-fixable): add a key projection.
  C high, B high, v3=0    -> LOCUS/INTEGRATION: read math works, transformer path
                             kills it (tanh/budget/residual/ln competition).
  C low                   -> deeper problem in the read formulation itself.

Usage:  python scripts/oracle_diag.py [--fast]
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
from oracle_probe_v3 import clean_keys


@torch.no_grad()
def offline_read_recall(emb, val_range, Kmat, Vtoks, query_mat):
    """M = Σ outer(emb[v_k], key_k). recall = argmax_v emb_v · (M @ query_k) == v_k.
    Kmat: (n,d) stored keys; query_mat: (n,d) query keys; Vtoks: value token ids."""
    n, d = Kmat.shape
    dmodel = emb.shape[1]
    M = torch.zeros(dmodel, d, device=emb.device)
    for i in range(n):
        M += torch.outer(emb[Vtoks[i]], Kmat[i])
    cand = emb[val_range]                       # (n_vals, d_model)
    hits = 0
    for i in range(n):
        r = M @ query_mat[i]                    # (d_model,)
        scores = cand @ r                       # (n_vals,)
        pred_val_token = val_range[int(scores.argmax())]
        hits += int(pred_val_token == Vtoks[i])
    return hits / max(n, 1)


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    task = AssociativeRecall(n_keys=16, n_vals=16, pairs_per_seq=4, seed=args.seed)
    cfg = SubstrateConfig(group_size=4)
    chance = 1.0 / task.n_vals
    mlog = MetricsLog(Path(__file__).parent.parent / "metrics.jsonl",
                      run_name=f"oracle-diag-seed{args.seed}",
                      config={"seed": args.seed, "chance": chance,
                              "arm": "RULINGS3_V2_orthogonality_diag"})

    torch.manual_seed(args.seed)
    model = TinyTransformer(task.vocab_size, d_model=args.d_model,
                            n_layers=args.layers, d_ff=args.d_ff, cfg=cfg,
                            use_substrate=True).to(device)
    print("--- pretrain ---")
    pretrain(model, task, mapping_seed_range=10_000, steps=args.pretrain_steps,
             device=device)

    ff_out = model.blocks[-1].ff_out
    mapping = task.random_mapping(task.held_keys, task.n_vals, 2)
    keyvec = clean_keys(model, ff_out, task, mapping, task.held_keys, device, args)

    keys = task.held_keys
    Kmat = torch.stack([keyvec[k] / (keyvec[k].norm() + 1e-8) for k in keys])
    Vtoks = [task.n_keys + mapping[k] for k in keys]
    val_range = list(range(task.n_keys, task.n_keys + task.n_vals))
    emb = model.tok.weight.detach()

    # A. key overlap
    G = Kmat @ Kmat.t()                         # cosine (rows unit-norm)
    n = G.shape[0]
    off = G[~torch.eye(n, dtype=torch.bool, device=device)].abs()
    print("\n[A] clean-key overlap (cosine):")
    print(f"    off-diagonal |cos|  mean {off.mean():.3f}  max {off.max():.3f}  "
          f"(0 = orthogonal/separable, ->1 = degenerate)")

    # B. offline read with real keys (store & query = real keys)
    recall_real = offline_read_recall(emb, val_range, Kmat, Vtoks, Kmat)
    print(f"\n[B] offline associative read, REAL keys: recall {recall_real:.3f} "
          f"(chance {chance:.3f})")

    # C. ceiling with orthonormal keys (QR of the real-key span)
    Q, _ = torch.linalg.qr(Kmat.t())            # (d_ff, n) orthonormal columns
    Eorth = Q.t()[:n]                           # (n, d_ff) orthonormal rows
    recall_orth = offline_read_recall(emb, val_range, Eorth, Vtoks, Eorth)
    print(f"[C] offline associative read, ORTHONORMAL keys (ceiling): "
          f"recall {recall_orth:.3f}")

    # Cross-check: decorrelated real keys (whiten via G^-1/2) read with themselves
    evals, evecs = torch.linalg.eigh(G + 1e-4 * torch.eye(n, device=device))
    Ginvhalf = evecs @ torch.diag(evals.clamp_min(1e-6).rsqrt()) @ evecs.t()
    Kw = Ginvhalf @ Kmat                        # decorrelated keys (n,d_ff)
    Kw = Kw / (Kw.norm(dim=-1, keepdim=True) + 1e-8)
    recall_white = offline_read_recall(emb, val_range, Kw, Vtoks, Kw)
    print(f"[C'] offline read, WHITENED real keys: recall {recall_white:.3f}")

    print("\n================ V2 VERDICT ================")
    v3 = args.v3_recall
    if recall_orth > 0.5 and recall_real < chance + 0.10:
        print(f"KEY-ORTHOGONALITY is the blocker (F4-FIXABLE). Mechanism works "
              f"with separable keys ([C] {recall_orth:.3f}); real keys overlap "
              f"([A] mean|cos| {off.mean():.3f}) so real-key read fails "
              f"([B] {recall_real:.3f}). -> add a learned key projection; do NOT "
              f"abandon the additive locus yet.")
    elif recall_orth > 0.5 and recall_real > 0.5:
        print(f"READ MATH WORKS on real keys offline ([B] {recall_real:.3f}) but "
              f"in-model v3 recall is {v3}. -> LOCUS/INTEGRATION: the transformer "
              f"downstream (tanh bound / budget / residual+ln competition) "
              f"destroys a retrievable signal. Fix integration, not keys.")
    else:
        print(f"Even orthonormal keys fail ([C] {recall_orth:.3f}) -> the read "
              f"FORMULATION itself is wrong; needs a dedicated read path, not a "
              f"weight perturbation. F4-FAIL (fundamental).")

    mlog.log(model="oracle_diag", phase="diag", key_cos_mean=float(off.mean()),
             key_cos_max=float(off.max()), offline_real=recall_real,
             offline_orthonormal=recall_orth, offline_whitened=recall_white,
             chance=chance)
    out = Path(__file__).parent.parent / "oracle_diag_results.json"
    out.write_text(json.dumps({"seed": args.seed, "chance": chance,
                               "key_cos_mean": float(off.mean()),
                               "key_cos_max": float(off.max()),
                               "offline_real": recall_real,
                               "offline_orthonormal": recall_orth,
                               "offline_whitened": recall_white}, indent=2))
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
    p.add_argument("--v3_recall", type=float, default=0.0,
                   help="the in-model oracle v3 recall for the verdict text")
    a = p.parse_args()
    if a.fast:
        a.d_model, a.layers, a.d_ff = 64, 2, 128
        a.pretrain_steps = 300
        a.oracle_batches = 3
    run(a)


if __name__ == "__main__":
    main()
