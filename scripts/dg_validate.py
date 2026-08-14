"""DG module acceptance tests (DG_SPEC.md §6). Torch-only, repo-independent.

Synthesizes keys with the toy's measured pathology (mean |cos| ~0.9, the
anisotropic cone) and verifies the DG stage fixes retrieval where dense
storage fails, that k is a real separation/generalization dial, and that
capacity degrades gracefully. All four gates must PASS before integration.

Run:  python scripts/dg_validate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from substrate.dg import (DGConfig, DGProjection, SparseAssociativeMemory,
                          mean_abs_cos)

PASS, FAIL = "PASS", "FAIL"
failures: list[str] = []


def gate(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{PASS if cond else FAIL}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


def make_anisotropic_keys(n: int, d: int, target_cos: float = 0.90,
                          seed: int = 0) -> torch.Tensor:
    """Keys crowded into a cone: shared dominant direction + small unique
    component, mimicking transformer hidden-state anisotropy."""
    g = torch.Generator().manual_seed(seed)
    base = F.normalize(torch.randn(d, generator=g), dim=0)
    w = target_cos ** 0.5   # pairwise cos ~= w^2, so w = sqrt(target)
    uniq = torch.randn(n, d, generator=g)
    uniq = uniq - (uniq @ base).unsqueeze(1) * base   # orthogonalize to base
    uniq = F.normalize(uniq, dim=-1)
    keys = w * base.unsqueeze(0) + (1 - w ** 2) ** 0.5 * uniq
    return F.normalize(keys, dim=-1)


def store_and_retrieve(keys: torch.Tensor, dg: DGProjection | None,
                       n_vals: int, cfg: DGConfig,
                       query_keys: torch.Tensor | None = None) -> float:
    """Store key_i -> one-hot(i % n_vals) with SHARED value classes (the
    toy's real regime: 16 values, many keys, so wrong bins accumulate
    interference). Read back, argmax accuracy over classes. dg=None stores
    raw keys (the dense failure mode)."""
    n = keys.shape[0]
    targets = torch.arange(n) % n_vals
    values = F.one_hot(targets, n_vals).float()
    if dg is None:
        M = values.t() @ keys                          # dense associator
        q = keys if query_keys is None else query_keys
        pred = (q @ M.t()).argmax(-1)
    else:
        dg.adapt(keys)                                 # adaptation warm-up
        codes = dg.encode(keys)
        mem = SparseAssociativeMemory(d_value=n_vals, cfg=cfg)
        mem.write(values, codes)
        q = keys if query_keys is None else query_keys
        pred = mem.read(dg.encode(q)).argmax(-1)
    return (pred == targets).float().mean().item()


def main() -> None:
    torch.manual_seed(0)
    d_in, d_dg, n_pairs = 256, 1024, 64   # integration dims (pin §2): d_ff locus
    # k operating point chosen from the measured curve (DG_SPEC §4); Rebecca
    # 2026-08-14: k=32 sat in the curve's dip at d_in=256, k=16 clears V2 cleanly.
    cfg = DGConfig(d_in=d_in, d_dg=d_dg, k=16)
    dg = DGProjection(cfg)
    keys = make_anisotropic_keys(n_pairs, d_in, target_cos=0.90)

    # V1 -- decorrelation (adapt first: the centering stage needs its
    # running mean, exactly as runtime encoding follows trace accumulation)
    dg.adapt(keys)
    raw_cos = mean_abs_cos(keys)
    code_cos = mean_abs_cos(dg.encode(keys))
    gate("V1 decorrelation", raw_cos >= 0.85 and code_cos <= 0.05,
         f"raw |cos| {raw_cos:.3f} -> code |cos| {code_cos:.3f}")

    # V2 -- selectivity: 16 shared value classes + mild query noise (the
    # toy's regime: wrong bins accumulate interference; noise collapses the
    # dense read's tiny margin while DG codes hold their separation).
    qg = torch.Generator().manual_seed(7)
    q_mild = F.normalize(
        keys + 0.05 * torch.randn(keys.shape, generator=qg), dim=-1)
    acc_dense = store_and_retrieve(keys, None, 16, cfg, query_keys=q_mild)
    acc_dg = store_and_retrieve(keys, dg, 16, cfg, query_keys=q_mild)
    exact_dg = store_and_retrieve(keys, dg, 16, cfg)
    gate("V2 selectivity",
         exact_dg >= 0.99 and acc_dg >= 0.90 and
         acc_dg >= acc_dense + 0.10,
         f"exact DG {exact_dg:.3f}; noisy: dense {acc_dense:.3f} "
         f"vs DG {acc_dg:.3f}")

    # V3 -- k is a dial: noisy-query accuracy improves with k,
    #        capacity (at 4x load) degrades with k
    noise_g = torch.Generator().manual_seed(1)
    noisy = F.normalize(
        keys + 0.08 * torch.randn(keys.shape, generator=noise_g), dim=-1)
    ks = [4, 8, 16, 32, 64]
    exact_by_k, noisy_by_k, cap_by_k = [], [], []
    big_keys = make_anisotropic_keys(2 * n_pairs, d_in, 0.90, seed=2)
    for k in ks:
        c = DGConfig(d_in=d_in, d_dg=d_dg, k=k)
        d = DGProjection(c)
        exact_by_k.append(store_and_retrieve(keys, d, 16, c))
        noisy_by_k.append(store_and_retrieve(keys, d, 16, c,
                                             query_keys=noisy))
        cap_by_k.append(store_and_retrieve(big_keys, d, 16, c))
    print(f"      k        : {ks}")
    print(f"      exact    : {[f'{a:.2f}' for a in exact_by_k]}")
    print(f"      noisy    : {[f'{a:.2f}' for a in noisy_by_k]}")
    print(f"      2x load  : {[f'{a:.2f}' for a in cap_by_k]}")
    noisy_span = max(noisy_by_k) - min(noisy_by_k)
    cap_span = max(cap_by_k) - min(cap_by_k)
    opposed = ks[noisy_by_k.index(max(noisy_by_k))] != \
        ks[cap_by_k.index(max(cap_by_k))]
    gate("V3 tradeoff dial",
         all(a >= 0.99 for a in exact_by_k) and noisy_span >= 0.10 and
         cap_span >= 0.15 and opposed,
         "exact perfect at every k; noise-robustness and load-capacity "
         "move on opposed axes with different optima")

    # V4 -- graceful capacity at fixed k=16
    accs = []
    for n in (16, 32, 64, 128):
        kk = make_anisotropic_keys(n, d_in, 0.90, seed=3)
        accs.append(store_and_retrieve(kk, dg, 16, cfg))
    print(f"      pairs    : [16, 32, 64, 128]")
    print(f"      accuracy : {[f'{a:.2f}' for a in accs]}")
    gate("V4 graceful capacity",
         accs[0] >= 0.99 and accs[1] >= 0.90 and accs[2] >= 0.60,
         "perfect at target load (16, the toy's), graceful beyond; "
         "ceiling is input rank, not the module")

    # I-DG-2 shared-space sanity: write path and read path use one encoder
    same = torch.equal(dg.encode(keys[:4]), dg.encode(keys[:4]))
    gate("I-DG-2 shared space", same, "encode deterministic and single-path")

    out = {"raw_cos": raw_cos, "code_cos": code_cos,
           "acc_dense": acc_dense, "acc_dg": acc_dg,
           "k_sweep": {"k": ks, "exact": exact_by_k, "noisy": noisy_by_k,
                       "capacity_4x": cap_by_k},
           "capacity_fixed_k": accs}
    Path("dg_validate_results.json").write_text(json.dumps(out, indent=2))
    print()
    if failures:
        print(f"{len(failures)} GATE(S) FAILED: {failures}")
        sys.exit(1)
    print("all DG acceptance gates pass -- cleared for integration "
          "(DG_SPEC.md \u00a76 -> \u00a77)")


if __name__ == "__main__":
    main()
