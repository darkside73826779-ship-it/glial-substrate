"""R6 positive controls (RULINGS_2 §2) — localize the failure to WRITE vs READ.

Two arms, pure experiment code, NO invariant changes:

1. Metric-sanity control (RULINGS_2 §2 second-tier): a model-independent
   exact-match key->value table answered at the probe position. Proves the
   probe harness itself CAN leave chance. Expect ~1.0. If it fails, the probe
   is broken and every 0.000 to date is meaningless.

2. Oracle-trace arm (RULINGS_2 §2 primary): bypass learned gating and Hebbian
   accumulation entirely; write into `fast` (through the normal budget-rescaled
   tick arithmetic, gate fully open) the IDEAL trace for each held-out pair —
   the outer product of the layer's post-activation at the VALUE position with
   its pre-activation at the KEY position, for the binding pair ONLY, nothing
   else (no smear from format tokens or other pairs). Then run the standard
   out-of-context probe.

   THE FORK:
     oracle recall  > chance  -> the READ path works; the gap is WRITE SELECTION
                                  (D-smear). Proceed to write-salience (R2).
     oracle recall == chance  -> even perfect writes are unreadable through
                                  alpha*tanh + downstream; the gap is RETRIEVAL
                                  (D-squash / D-faint). Sweep alpha in the ORACLE
                                  ARM ONLY (diagnostic; never a learning run) and
                                  a linear-read variant; if readability appears
                                  only at higher alpha / linear read, that is the
                                  evidence package for a considered I-1 revision
                                  (human sign-off required; bounds are guarantees).

Usage:
  python scripts/oracle_probe.py            # full
  python scripts/oracle_probe.py --fast     # smoke-scale
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


# --------------------------------------------------------------------------- #
# Arm 1 — metric-sanity control (model-independent)                            #
# --------------------------------------------------------------------------- #

class KVOracleModel:
    """Answers the recall probe from an exact table, ignoring learned weights.
    At every position it emits a one-hot logit for n_keys+mapping[token] when
    the input token is a key. recall_acc masks to the answer position (input =
    query key), so this returns the true value there. Validates the harness."""

    def __init__(self, task, mapping, device):
        self.task = task
        self.mapping = mapping
        self.device = device
        self.V = task.vocab_size

    def __call__(self, X):
        B, T = X.shape
        logits = torch.full((B, T, self.V), -10.0, device=self.device)
        for b in range(B):
            for t in range(T):
                tok = int(X[b, t])
                if tok in self.mapping:
                    logits[b, t, self.task.n_keys + self.mapping[tok]] = 10.0
        return logits

    def eval(self):
        return self


# --------------------------------------------------------------------------- #
# Arm 2 — oracle-trace write                                                   #
# --------------------------------------------------------------------------- #

def build_binding_batch(task, mapping, keys, k, device, batch, n_pairs):
    """Sequences that SHOW the binding pair (k, val) in-context and END with
    [SEP, QUERY, k] — so at the final position the model is querying k and (pair
    shown) emits the correct value in-context. We bind at that final QUERY
    position: pre = FFN input there, post = FFN output there. This is the
    substrate's own same-position Hebbian rule outer(post, pre), keyed on
    'querying k -> emit value', which is exactly what retrieval needs. Returns
    X and the (uniform) final query position index."""
    seqs = []
    others = [x for x in keys if x != k]
    for _ in range(batch):
        fill = torch.tensor(others)[torch.randperm(len(others))][
            : n_pairs - 1].tolist()
        slot = int(torch.randint(n_pairs, (1,)))
        ks = fill[:slot] + [k] + fill[slot:]   # binding pair shown in-context
        seq = []
        for kk in ks:
            seq += [kk, task.n_keys + mapping[kk]]
        seq += [task.SEP, task.QUERY, k]        # trailing query on k
        seqs.append(torch.tensor(seq))
    X = torch.stack(seqs).to(device)
    final_pos = X.shape[1] - 1                   # uniform length -> last token
    return X, final_pos


@torch.no_grad()
def build_oracle_fast(model, task, mapping, keys, device, args, alpha):
    """Accumulate, per substrate layer, the ideal binding outer products
    outer(post_value, pre_key) over sampled contexts — binding pair only —
    then convert to `fast` with gate fully open and the standard per-group
    budget rescale at the given alpha. Returns {layer: fast_tensor}."""
    layers = model.substrate_layers()
    accum = {l: torch.zeros(l.out_features, l.in_features, device=device)
             for l in layers}
    counts = {l: 0 for l in layers}

    for k in keys:
        for _ in range(args.oracle_batches):
            X, fpos = build_binding_batch(
                task, mapping, keys, k, device, args.batch, task.pairs_per_seq)
            act = {}
            fast_map = {l: torch.zeros(l.out_features, l.in_features,
                                       device=device) for l in layers}
            _ = model(X, fast_map=fast_map, act_out=act)  # W_base forward; caches
            for l in layers:
                pre, post = act[l]                        # (B,T,in), (B,T,out)
                B = pre.shape[0]
                for b in range(B):
                    p = pre[b, fpos]                      # (in,)  query-pos input
                    q = post[b, fpos]                     # (out,) value-emit output
                    p = p / (p.norm() + 1e-8)
                    q = q / (q.norm() + 1e-8)
                    accum[l] += torch.outer(q, p)
                    counts[l] += 1

    oracle_fast = {}
    for l in layers:
        trace = accum[l] / max(counts[l], 1)              # ideal, smear-free
        lr = l.log_plastic_lr.exp().item()
        fast = lr * trace                                 # gate == 1, consol == 0
        # budget rescale at THIS alpha (mirror of _budget_rescale, alpha-param)
        contrib = alpha * torch.tanh(fast)
        gv = fast.view(l.out_features, l.n_groups, l.cfg.group_size)
        gnorm = contrib.view(l.out_features, l.n_groups, l.cfg.group_size).norm(dim=-1)
        bnorm = l.W_base.view(l.out_features, l.n_groups, l.cfg.group_size).norm(dim=-1)
        limit = l.cfg.budget_frac * bnorm + 1e-8
        scale = torch.clamp(limit / (gnorm + 1e-8), max=1.0)
        fast = (gv * scale.unsqueeze(-1)).reshape_as(fast)
        oracle_fast[l] = fast
    return oracle_fast


@torch.no_grad()
def eval_oracle(model, task, mapping, keys, device, args, alpha,
                linear_scale=1.0):
    """Install oracle fast at `alpha`, run the out-of-context probe.
    linear_scale<1 shrinks fast toward the tanh linear regime (magnitude
    compensated by alpha), the RULINGS_2 linear-read diagnostic."""
    layers = model.substrate_layers()
    oracle_fast = build_oracle_fast(model, task, mapping, keys, device, args,
                                    alpha)
    saved_alpha = {l: l.cfg.alpha for l in layers}
    model.set_mode("runtime")
    for l in layers:
        l.cfg.alpha = alpha
        f = oracle_fast[l] * linear_scale
        l.fast.zero_()
        l.fast.add_(f)
    # out-of-context: queried pair absent from the probe sequence
    acc = recall_acc(model, task, mapping, keys, include_query_pair=False,
                     device=device)
    in_ctx = recall_acc(model, task, mapping, keys, include_query_pair=True,
                        device=device)
    for l in layers:                                       # restore
        l.cfg.alpha = saved_alpha[l]
        l.reset_substrate()
    return acc, in_ctx


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    task = AssociativeRecall(n_keys=16, n_vals=16, pairs_per_seq=4, seed=args.seed)
    cfg = SubstrateConfig(group_size=4)
    chance = 1.0 / task.n_vals

    mlog = MetricsLog(Path(__file__).parent.parent / "metrics.jsonl",
                      run_name=f"oracle-probe-seed{args.seed}",
                      config={"seed": args.seed, "chance": chance,
                              "arm": "R6_oracle_positive_control"})

    torch.manual_seed(args.seed)
    model = TinyTransformer(task.vocab_size, d_model=args.d_model,
                            n_layers=args.layers, d_ff=args.d_ff, cfg=cfg,
                            use_substrate=True).to(device)
    print("--- pretrain (format acquisition) ---")
    pretrain(model, task, mapping_seed_range=10_000, steps=args.pretrain_steps,
             device=device)

    rt_mapping = task.random_mapping(task.held_keys, task.n_vals, 2)

    # Arm 1: metric-sanity control.
    kv = KVOracleModel(task, rt_mapping, device)
    kv_acc = recall_acc(kv, task, rt_mapping, task.held_keys,
                        include_query_pair=False, device=device)
    print(f"\n[metric-sanity] KV-oracle out-of-context recall: {kv_acc:.3f} "
          f"(expect ~1.0; chance {chance:.3f})")
    mlog.log(model="kv_oracle", phase="control", when="metric_sanity",
             out_of_context_heldout=kv_acc)

    # Arm 2: oracle-trace, alpha sweep + linear-read variant.
    print("\n[oracle-trace] ideal smear-free writes, out-of-context probe:")
    results = {"seed": args.seed, "chance": chance, "kv_sanity": kv_acc,
               "oracle": {}}
    for alpha in args.alphas:
        acc, in_ctx = eval_oracle(model, task, rt_mapping, task.held_keys,
                                  device, args, alpha)
        print(f"  alpha={alpha:<4}  oo-ctx {acc:.3f}   in-ctx {in_ctx:.3f}   "
              f"(chance {chance:.3f})")
        mlog.log(model="oracle", phase="control", alpha=alpha,
                 out_of_context_heldout=acc, in_context_heldout_pool=in_ctx)
        results["oracle"][f"alpha_{alpha}"] = {"oo_ctx": acc, "in_ctx": in_ctx}

    # linear-read diagnostic: small fast (near-linear tanh), alpha compensates.
    acc_lin, in_lin = eval_oracle(model, task, rt_mapping, task.held_keys,
                                  device, args, alpha=max(args.alphas),
                                  linear_scale=args.linear_scale)
    print(f"  linear-read (scale={args.linear_scale}, alpha={max(args.alphas)}): "
          f"oo-ctx {acc_lin:.3f}   in-ctx {in_lin:.3f}")
    mlog.log(model="oracle", phase="control", alpha=max(args.alphas),
             linear_scale=args.linear_scale, out_of_context_heldout=acc_lin,
             in_context_heldout_pool=in_lin)
    results["oracle"]["linear_read"] = {"oo_ctx": acc_lin, "in_ctx": in_lin}

    # Fork verdict.
    best = max(v["oo_ctx"] for v in results["oracle"].values())
    print("\n================ FORK VERDICT ================")
    print(f"KV metric-sanity: {kv_acc:.3f}  ->  "
          f"{'harness VALID' if kv_acc > 0.5 else 'HARNESS BROKEN — stop'}")
    print(f"best oracle oo-ctx: {best:.3f}  (chance {chance:.3f})")
    if best > chance + 0.10:
        print("READ path works. Gap = WRITE SELECTION (D-smear). "
              "-> build write-salience (RULINGS_2 R2).")
    else:
        print("Even perfect writes unreadable. Gap = RETRIEVAL "
              "(D-squash/D-faint). -> alpha/linear evidence package for I-1 "
              "review (human sign-off).")
    out = Path(__file__).parent.parent / "oracle_results.json"
    out.write_text(json.dumps(results, indent=2))
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
    p.add_argument("--oracle_batches", type=int, default=4)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.2, 0.4])
    p.add_argument("--linear_scale", type=float, default=0.15)
    a = p.parse_args()
    if a.fast:
        a.d_model, a.layers, a.d_ff = 64, 2, 128
        a.pretrain_steps = 300
        a.oracle_batches = 2
    run(a)


if __name__ == "__main__":
    main()
