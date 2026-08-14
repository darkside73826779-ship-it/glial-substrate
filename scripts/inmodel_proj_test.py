"""Step 1 (RULINGS PROGRAM_REVIEW P4): in-model fixed-projection confirmation.

Does the offline whitening result (1.000) survive the full transformer forward?
We fold a FIXED ZCA whitening of the clean-key subspace into the stored fast
matrix at the last block's ff_out, install it, and measure IN-MODEL:
  - out-of-context recall (acquisition signal),
  - pretrain-pool retention SIMULTANEOUSLY (P4 insertion 1),
  - raw match-strength signal, uncalibrated (P4 insertion 2),
across alpha and budget {0.25, off}. Diagnostic only: fixed projection (not yet
the learned meta-trained P), bounds swept in the oracle arm and restored, no
learning run, no invariant change.

Read folding: with clean keys k and values v, whitened store+read computes
  score_v(g') = emb_v . (M+ @ g'),   M+ = Σ_k outer(emb_v_k, C^+ key_k),
  C = (1/n) Σ_k keyhat_k keyhat_k^T  (d_ff), C^+ = regularized pseudo-inverse.
So the in-model read selects on key_k^T C^+ g' (whitened similarity). If the
offline sanity ~1.0 but in-model < that, the transformer path (tanh/budget/
residual/ln/head competition) is the remaining cost.

Verdict vs pre-registered acquisition bar (P1.A.1): recall >= 0.50 = fix
survives in-model; chance-level = the projection alone is insufficient in-model
(escalate to projection+sharpening, P2).

Usage:  python scripts/inmodel_proj_test.py [--fast]
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
from run_experiment import pretrain, recall_acc
from oracle_probe_v3 import clean_keys, budget_rescale


@torch.no_grad()
def pinv_sqrt_metric(Kmat, tol=1e-3):
    """C^+ for C = (1/n) Σ keyhat keyhat^T (d_ff), via eigendecomposition on the
    key subspace (rank <= n). Returns the d_ff x d_ff whitening metric C^+."""
    n, d = Kmat.shape
    C = (Kmat.t() @ Kmat) / n                     # (d_ff, d_ff), rank <= n
    evals, evecs = torch.linalg.eigh(C)
    keep = evals > (tol * evals.max())
    inv = torch.zeros_like(evals)
    inv[keep] = 1.0 / evals[keep]
    return evecs @ torch.diag(inv) @ evecs.t()    # C^+


@torch.no_grad()
def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    task = AssociativeRecall(n_keys=16, n_vals=16, pairs_per_seq=4, seed=args.seed)
    cfg = SubstrateConfig(group_size=4)
    chance = 1.0 / task.n_vals
    mlog = MetricsLog(Path(__file__).parent.parent / "metrics.jsonl",
                      run_name=f"inmodel-proj-seed{args.seed}",
                      config={"seed": args.seed, "chance": chance,
                              "arm": "PROGRAM_step1_inmodel_fixed_projection"})

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
    pre_mapping = task.random_mapping(task.train_keys, task.n_vals, 1)
    keyvec = clean_keys(model, ff_out, task, mapping, task.held_keys, device, args)

    keys = task.held_keys
    Kmat = torch.stack([keyvec[k] / (keyvec[k].norm() + 1e-8) for k in keys])
    emb = model.tok.weight.detach()
    val_range = list(range(task.n_keys, task.n_keys + task.n_vals))
    Cpinv = pinv_sqrt_metric(Kmat)

    # Projected store: M+ = Σ outer(emb_v_k, C^+ keyhat_k)
    Mproj = torch.zeros(ff_out.out_features, ff_out.in_features, device=device)
    for i, k in enumerate(keys):
        vt = task.n_keys + mapping[k]
        Mproj += torch.outer(emb[vt] / (emb[vt].norm() + 1e-8),
                             Cpinv @ Kmat[i])
    Mproj = Mproj / Mproj.norm()                   # unit Frobenius; alpha scales dose

    # Offline sanity: does the folded projection retrieve in isolation?
    cand = emb[val_range]
    hits = 0
    for i, k in enumerate(keys):
        r = Mproj @ Kmat[i]
        hits += int(val_range[int((cand @ r).argmax())] == task.n_keys + mapping[k])
    offline = hits / len(keys)
    print(f"\n[offline sanity] projected read recall: {offline:.3f} "
          f"(expect ~1.0 if folding correct; chance {chance:.3f})")

    print(f"\nin-model projected read (last ff_out). chance {chance:.3f}, "
          f"acquisition bar >= 0.50\n")
    print(f"{'budget':>7} {'alpha':>6} {'oo_recall':>10} {'retention':>10} "
          f"{'forget':>7} {'dose':>7}")
    layers = model.substrate_layers()
    grid = []
    for bf in ["off", "0.25"]:
        bfv = None if bf == "off" else float(bf)
        for alpha in args.alphas:
            fast = budget_rescale(ff_out, Mproj.clone(), bfv, alpha)
            saved_alpha = ff_out.cfg.alpha
            ff_out.cfg.alpha = alpha
            model.set_mode("runtime")
            for l in layers:
                l.reset_substrate()
            # retention baseline (no write)
            base_pre = recall_acc(model, task, pre_mapping, task.train_keys,
                                  device=device)
            ff_out.fast.copy_(fast)
            dW = ff_out.effective_weight() - ff_out.W_base
            dose = (dW.norm() / (ff_out.W_base.norm() + 1e-8)).item()
            oo = recall_acc(model, task, mapping, keys,
                            include_query_pair=False, device=device)
            ret = recall_acc(model, task, pre_mapping, task.train_keys,
                             device=device)
            forget = base_pre - ret
            print(f"{bf:>7} {alpha:>6} {oo:>10.3f} {ret:>10.3f} "
                  f"{forget:>7.3f} {dose:>7.3f}")
            mlog.log(model="inmodel_proj", phase="proj", budget=bf, alpha=alpha,
                     out_of_context_heldout=oo, retention=ret, forget=forget,
                     delivered_dose=dose)
            grid.append({"budget": bf, "alpha": alpha, "oo": oo, "ret": ret,
                         "forget": forget, "dose": dose})
            ff_out.cfg.alpha = saved_alpha
            for l in layers:
                l.reset_substrate()

    best = max(g["oo"] for g in grid)
    print("\n================ STEP 1 VERDICT ================")
    print(f"offline sanity: {offline:.3f}   best in-model oo-recall: {best:.3f} "
          f"(bar 0.50, chance {chance:.3f})")
    if best >= 0.50:
        print("Fix SURVIVES in-model. Projection read works through the "
              "transformer. -> build the LEARNED meta-trained P (P2a) + step 2 "
              "generalization gate.")
    elif best > chance + 0.10:
        print("Partial survival: projection helps in-model but below bar. -> "
              "escalate to projection+sharpening (Hopfield read, P2 escalation) "
              "before step 2.")
    else:
        print("Projection alone does NOT survive in-model. -> escalate to "
              "projection+sharpening (P2), the second and final Gap-1 attempt.")
    out = Path(__file__).parent.parent / "inmodel_proj_results.json"
    out.write_text(json.dumps({"seed": args.seed, "chance": chance,
                               "offline_sanity": offline, "grid": grid}, indent=2))
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
    p.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.4, 1.0])
    a = p.parse_args()
    if a.fast:
        a.d_model, a.layers, a.d_ff = 64, 2, 128
        a.pretrain_steps = 300
        a.oracle_batches = 3
    run(a)


if __name__ == "__main__":
    main()
