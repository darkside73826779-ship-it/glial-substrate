"""Operating-point curve for V2 at the integration dims (DG_SPEC §4).
Reports DG-vs-dense noise margin across k at d_in=256, d_dg=1024, so the
k operating point is chosen from the measured curve, not assumed. Diagnostic."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from substrate.dg import DGConfig, DGProjection
from dg_validate import make_anisotropic_keys, store_and_retrieve

torch.manual_seed(0)
d_in, d_dg, n_pairs = 256, 1024, 64
keys = make_anisotropic_keys(n_pairs, d_in, 0.90)
qg = torch.Generator().manual_seed(7)
q_mild = F.normalize(keys + 0.05 * torch.randn(keys.shape, generator=qg), dim=-1)
dense = store_and_retrieve(keys, None, 16, DGConfig(d_in=d_in, d_dg=d_dg), q_mild)
print(f"dense (no DG) noisy acc: {dense:.3f}")
print(f"{'k':>5} {'exactDG':>8} {'noisyDG':>8} {'margin':>8} {'>=0.10?':>8}")
for k in [4, 8, 16, 24, 32, 48, 64]:
    c = DGConfig(d_in=d_in, d_dg=d_dg, k=k)
    d = DGProjection(c)
    ex = store_and_retrieve(keys, d, 16, c)
    ny = store_and_retrieve(keys, d, 16, c, query_keys=q_mild)
    m = ny - dense
    print(f"{k:>5} {ex:>8.3f} {ny:>8.3f} {m:>8.3f} {'yes' if m>=0.10 else 'no':>8}")
