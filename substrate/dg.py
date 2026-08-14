"""Dentate gyrus module: pattern separation for the glial substrate.

Biology -> tensor (see DG_SPEC.md):
    expansion   EC->DG          z = P h        (random or meta-trained)
    inhibition  basket cells    s = kWTA(z, k) (top-k, graded, post-ReLU)
    detonation  mossy fibers    write gain     (existing gate machinery)
    storage     CA3             M += outer(v, s);  read: M s

Self-contained: torch only. Integration contract in DG_SPEC.md §7.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DGConfig:
    d_in: int = 128
    d_dg: int = 512          # expansion ratio ~4x (biology: 5x)
    k: int = 16              # active units (~3%; biology: 1-2%)
    mode: str = "random"     # "random" (R2, buffer) | "learned" (R3, parameter)
    write_norm: bool = True  # L2-normalize sparse code before write/read
    adapt_rate: float = 0.05 # EMA rate of the input-mean adaptation stage
    beta: float = 2.0        # per-column budget scale for M (I-DG-4)
    seed: int = 0


def kwta(z: torch.Tensor, k: int, hard: bool = True,
         temp: float = 0.1) -> torch.Tensor:
    """k-winners-take-all. hard=True: exact top-k of relu(z), graded values
    (runtime; non-differentiable). hard=False: soft mask via sigmoid around
    the k-th value (meta-training straight-through aid; functional path only
    per DG_SPEC I-DG note)."""
    a = F.relu(z)
    if k >= a.shape[-1]:
        return a
    thresh = a.topk(k, dim=-1).values[..., -1:]
    if hard:
        return a * (a >= thresh).to(a.dtype)
    soft = torch.sigmoid((a - thresh) / max(temp, 1e-6))
    hard_mask = (a >= thresh).to(a.dtype)
    return a * (hard_mask + soft - soft.detach())   # straight-through


class DGProjection(nn.Module):
    """Adaptation + expansion + sparsification. In 'random' mode P is a frozen
    buffer (zero learnable parameters -> cannot be a copy machine). In
    'learned' mode P is a parameter intended for outer-loop meta-training
    only and must be frozen at runtime (I-DG-1).

    Centering stage (feedforward inhibition / adaptation): anisotropic keys
    share a dominant cone direction; expansion alone maps that shared
    component to the same winners for every key. Subtracting a running mean
    of recent inputs removes the common mode before projection -- a local,
    forward-only statistic (EMA buffer), never per-key fitting."""

    def __init__(self, cfg: DGConfig):
        super().__init__()
        self.cfg = cfg
        g = torch.Generator().manual_seed(cfg.seed)
        P = torch.randn(cfg.d_dg, cfg.d_in, generator=g) / (cfg.d_in ** 0.5)
        if cfg.mode == "random":
            self.register_buffer("P", P)
        else:
            self.P = nn.Parameter(P)
        self.register_buffer("mu", torch.zeros(cfg.d_in))
        self.register_buffer("mu_initialized", torch.zeros(1))

    @torch.no_grad()
    def adapt(self, h: torch.Tensor) -> None:
        """Update the running input mean (the adaptation state). Call from
        the same sites that accumulate traces; forward-only, local (I-5)."""
        m = h.detach().reshape(-1, self.cfg.d_in).mean(0)
        if self.mu_initialized.item() == 0:
            self.mu.copy_(m)
            self.mu_initialized.fill_(1)
        else:
            self.mu.mul_(1 - self.cfg.adapt_rate).add_(
                m, alpha=self.cfg.adapt_rate)

    def encode(self, h: torch.Tensor, hard: bool = True) -> torch.Tensor:
        """h: (..., d_in) -> sparse code (..., d_dg). Same path serves write
        and read (I-DG-2); callers must not duplicate this logic."""
        s = kwta((h - self.mu) @ self.P.t(), self.cfg.k, hard=hard)
        if self.cfg.write_norm:
            s = F.normalize(s, dim=-1)
        return s


class SparseAssociativeMemory(nn.Module):
    """CA3 analog: linear hetero-associator over DG codes.

    Runtime state M is a buffer (never a Parameter; I-DG-1) and in the full
    system mutates only at tick (I-6) -- write() here is the primitive the
    tick calls. Per-column budget bounds M's contribution (I-DG-4).
    """

    def __init__(self, d_value: int, cfg: DGConfig):
        super().__init__()
        self.cfg = cfg
        self.d_value = d_value
        self.register_buffer("M", torch.zeros(d_value, cfg.d_dg))
        self.register_buffer("col_budget",
                             torch.full((cfg.d_dg,), cfg.beta))

    @torch.no_grad()
    def write(self, value: torch.Tensor, code: torch.Tensor,
              gain: float = 1.0) -> None:
        """value: (d_value,) or (B, d_value); code: matching (d_dg,)/(B, d_dg)."""
        v = value.reshape(-1, self.d_value)
        s = code.reshape(-1, self.cfg.d_dg)
        self.M += gain * (v.t() @ s)
        self._enforce_budget()

    @torch.no_grad()
    def _enforce_budget(self) -> None:
        norms = self.M.norm(dim=0)                       # per DG unit/column
        scale = torch.clamp(self.col_budget / (norms + 1e-8), max=1.0)
        self.M *= scale.unsqueeze(0)

    def read(self, code: torch.Tensor) -> torch.Tensor:
        """Linear read: (..., d_dg) -> (..., d_value). Selective iff codes
        are decorrelated -- that is the whole point of the DG stage."""
        return code @ self.M.t()

    def read_completed(self, code: torch.Tensor, stored_codes: torch.Tensor,
                       stored_values: torch.Tensor,
                       temp: float = 0.05) -> torch.Tensor:
        """Escalation stage (DG_SPEC §4): separation-then-completion.
        Softmax over similarity to stored codes = one-step Hopfield/attention
        cleanup. Only engaged if the linear read fails the gates."""
        att = torch.softmax(code @ stored_codes.t() / temp, dim=-1)
        return att @ stored_values

    @torch.no_grad()
    def reset(self) -> None:
        self.M.zero_()


def mean_abs_cos(x: torch.Tensor) -> float:
    """Mean pairwise |cosine| of row vectors -- the collinearity metric
    (the toy measured 0.914 on raw keys; DG codes should land near 0)."""
    xn = F.normalize(x, dim=-1)
    c = xn @ xn.t()
    n = c.shape[0]
    off = c.masked_select(~torch.eye(n, dtype=torch.bool, device=c.device))
    return off.abs().mean().item()
