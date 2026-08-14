"""Full experiment: substrate model vs parameter-matched frozen baseline.

Phases:
  1. PRETRAIN (backprop, both models identically): learn the associative
     recall FORMAT on the pretrain key pool.
  2. RUNTIME EXPOSURE (no gradients anywhere): stream sequences using
     HELD-OUT keys with a fixed novel mapping. Substrate model accumulates
     traces and ticks; baseline just runs forward.
  3. PROBE: out-of-context recall — query a held-out pair WITHOUT showing
     it in the sequence. Only persistent runtime memory can answer.
     Also re-measure pretrain-pool accuracy (forgetting check).

Usage:
  python scripts/run_experiment.py            # full run
  python scripts/run_experiment.py --fast     # reduced steps (smoke-scale)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from substrate import (AssociativeRecall, SubstrateConfig,
                       SubstrateController, TinyTransformer)
from substrate.metrics import MetricsLog


def loss_fn(logits, targets):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           targets.reshape(-1), ignore_index=-100)


@torch.no_grad()
def recall_acc(model, task, mapping, keys, batches=20, batch=32,
               include_query_pair=True, device="cpu"):
    hits = total = 0
    for _ in range(batches):
        X, T, _ = task.make_batch(batch, mapping, keys, device,
                                  include_query_pair=include_query_pair)
        logits = model(X)
        mask = T != -100
        pred = logits.argmax(-1)
        hits += (pred[mask] == T[mask]).sum().item()
        total += mask.sum().item()
    return hits / max(total, 1)


def pretrain(model, task, mapping_seed_range, steps, device, lr=3e-4,
             mlog=None, model_name=""):
    model.set_mode("pretrain")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr)
    for step in range(steps):
        seed = int(torch.randint(mapping_seed_range, (1,)))
        mapping = task.random_mapping(task.train_keys, task.n_vals, seed)
        X, T, _ = task.make_batch(32, mapping, task.train_keys, device)
        loss = loss_fn(model(X), T)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if mlog is not None and step % 5 == 0:
            mlog.log(model=model_name, phase="pretrain", loss=loss.item())
        if step % max(steps // 10, 1) == 0:
            print(f"  pretrain step {step:5d}  loss {loss.item():.4f}")
    return model


def runtime_exposure(model, task, mapping, steps, device, substrate: bool,
                     tick_every=8, mlog=None, model_name=""):
    """Stream held-out-pool sequences. No gradients. Substrate ticks."""
    model.eval()
    if substrate:
        model.set_mode("runtime")
        ctrl = SubstrateController(model, tick_every=tick_every)
    with torch.no_grad():
        for step in range(steps):
            X, T, _ = task.make_batch(16, mapping, task.held_keys, device)
            logits = model(X)
            loss = loss_fn(logits, T).item()
            row = {"model": model_name, "phase": "exposure", "loss": loss}
            if substrate:
                ctrl.observe_loss(loss)
                stats = ctrl.maybe_tick()
                if stats:
                    s = stats[0]
                    row.update(gate_mean=s["gate_mean"],
                               fast_abs_mean=s["fast_abs_mean"],
                               consol_mean=s["consol_mean"],
                               neuromod=ctrl.neuromod())
                if stats and step % max(steps // 5, 1) < tick_every:
                    s = stats[0]
                    print(f"  exposure step {step:4d}  loss {loss:.3f}  "
                          f"gate {s['gate_mean']:.3f}  "
                          f"|fast| {s['fast_abs_mean']:.4f}  "
                          f"consol {s['consol_mean']:.4f}")
            if mlog is not None:
                mlog.log(**row)


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    task = AssociativeRecall(n_keys=16, n_vals=16, pairs_per_seq=4,
                             seed=args.seed)
    cfg = SubstrateConfig(group_size=4)
    mlog = MetricsLog(Path(__file__).parent.parent / "metrics.jsonl",
                      run_name=f"assoc-recall-seed{args.seed}",
                      config={"seed": args.seed, "d_model": args.d_model,
                              "layers": args.layers, "d_ff": args.d_ff,
                              "chance": 1.0 / task.n_vals,
                              "pretrain_steps": args.pretrain_steps,
                              "exposure_steps": args.exposure_steps})

    results = {}
    for name, use_sub in [("substrate", True), ("baseline", False)]:
        print(f"\n=== {name} ===")
        mlog.phase(f"{name}:pretrain")
        torch.manual_seed(args.seed)  # identical init for both
        model = TinyTransformer(task.vocab_size, d_model=args.d_model,
                                n_layers=args.layers, d_ff=args.d_ff,
                                cfg=cfg, use_substrate=use_sub).to(device)
        t0 = time.time()
        pretrain(model, task, mapping_seed_range=10_000,
                 steps=args.pretrain_steps, device=device,
                 mlog=mlog, model_name=name)

        pre_mapping = task.random_mapping(task.train_keys, task.n_vals, 1)
        rt_mapping = task.random_mapping(task.held_keys, task.n_vals, 2)

        before = {
            "in_context_pretrain_pool": recall_acc(
                model, task, pre_mapping, task.train_keys, device=device),
            "in_context_heldout_pool": recall_acc(
                model, task, rt_mapping, task.held_keys, device=device),
            "out_of_context_heldout": recall_acc(
                model, task, rt_mapping, task.held_keys,
                include_query_pair=False, device=device),
        }
        print(f"  before exposure: {json.dumps(before, indent=None)}")
        mlog.log(model=name, phase="probe", when="before", **before)

        mlog.phase(f"{name}:exposure")
        runtime_exposure(model, task, rt_mapping, args.exposure_steps,
                         device, substrate=use_sub, tick_every=args.tick_every,
                         mlog=mlog, model_name=name)

        after = {
            "in_context_pretrain_pool": recall_acc(
                model, task, pre_mapping, task.train_keys, device=device),
            "in_context_heldout_pool": recall_acc(
                model, task, rt_mapping, task.held_keys, device=device),
            "out_of_context_heldout": recall_acc(
                model, task, rt_mapping, task.held_keys,
                include_query_pair=False, device=device),
        }
        print(f"  after exposure:  {json.dumps(after, indent=None)}")
        mlog.log(model=name, phase="probe", when="after", **after)
        results[name] = {"before": before, "after": after,
                         "seconds": round(time.time() - t0, 1)}

    print("\n=== summary ===")
    chance = 1.0 / task.n_vals
    print(f"  chance level: {chance:.3f}")
    for name, r in results.items():
        gain = r["after"]["out_of_context_heldout"] - \
            r["before"]["out_of_context_heldout"]
        forget = r["before"]["in_context_pretrain_pool"] - \
            r["after"]["in_context_pretrain_pool"]
        print(f"  {name:10s} out-of-context recall gain: {gain:+.3f}   "
              f"pretrain-pool forgetting: {forget:+.3f}")
    out = Path(__file__).parent.parent / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"  written: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fast", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=256)
    p.add_argument("--pretrain_steps", type=int, default=3000)
    p.add_argument("--exposure_steps", type=int, default=800)
    p.add_argument("--tick_every", type=int, default=8)
    a = p.parse_args()
    if a.fast:
        a.pretrain_steps, a.exposure_steps = 300, 120
        a.d_model, a.layers, a.d_ff = 64, 2, 128
    run(a)
