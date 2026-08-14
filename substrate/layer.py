"""SubstrateLinear: a linear layer with a living sidecar substrate.

Mapping to the design conversation:
  - W_base          : slow synaptic weights (trained by backprop, frozen at runtime)
  - fast            : fast-weight state written by local activity (the loop-back memory)
  - trace           : eligibility trace, Hebbian outer(post, pre), EMA-decayed
  - consol          : per-weight consolidation importance (EWC-like, but live)
  - groups          : blocks of `group_size` weights along fan-in = one "glial unit"
                      (dendritic-branch analogy; functional neighborhood, NOT
                      positional adjacency of the raw matrix — see DESIGN.md I-3)
  - gate_net        : learned plasticity gate per group (the neuromodulation hook)
  - budget          : per-group heterosynaptic strength budget (stability)

Two timescales (DESIGN.md I-6):
  forward()  — pure matmul over effective weights; cheap; also accumulates
               traces under no_grad in runtime mode.
  tick()     — the slow pass; the ONLY place fast/consol mutate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SubstrateConfig:
    group_size: int = 4          # weights per glial unit, along fan-in
    alpha: float = 0.10          # max fast-weight contribution scale (I-1)
    budget_frac: float = 0.25    # per-group ||fast_contrib|| <= frac * ||W_base|| (I-2)
    trace_decay: float = 0.90    # EMA decay of eligibility per forward
    fast_decay: float = 0.995    # passive decay of fast weights per tick
    consol_decay: float = 0.999  # slow decay of consolidation per tick
    consol_rate: float = 0.05    # how fast repeated gated updates consolidate
    plastic_lr: float = 0.5      # runtime learning rate applied at tick
    gate_hidden: int = 16        # hidden width of the shared gate MLP
    gate_features: int = 5       # [mean|trace|, std(trace), mean|fast|, mean(consol), neuromod]
    trace_updates_per_tick_cap: int = 10_000  # sanity ceiling, not a tuning knob


class GateNet(nn.Module):
    """Tiny MLP shared across all groups of a layer: group stats -> gate in (0,1).

    This is the learned plasticity rule. Meta-training shapes THESE parameters
    with backprop (outer loop); runtime never does (I-4).
    Bias init is negative so plasticity starts mostly OFF (biology keeps
    plasticity off by default; see conversation on neuromodulatory gating).
    """

    def __init__(self, cfg: SubstrateConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.gate_features, cfg.gate_hidden),
            nn.Tanh(),
            nn.Linear(cfg.gate_hidden, 1),
        )
        with torch.no_grad():
            self.net[-1].bias.fill_(-1.0)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(feats)).squeeze(-1)


class SubstrateLinear(nn.Module):
    """Linear layer with frozen-at-runtime base weights and a plastic substrate.

    Modes:
      pretrain : W_base trains by backprop; substrate inert (fast stays 0).
      runtime  : W_base frozen; traces accumulate on forward; tick() mutates
                 fast/consol from local signals only. No gradients anywhere.
      meta     : functional inner-loop; see functional_tick() — gradients flow
                 to GateNet / meta scalars, never into runtime state buffers.
    """

    def __init__(self, in_features: int, out_features: int,
                 cfg: SubstrateConfig | None = None, bias: bool = True):
        super().__init__()
        self.cfg = cfg or SubstrateConfig()
        assert in_features % self.cfg.group_size == 0, (
            f"in_features={in_features} must be divisible by "
            f"group_size={self.cfg.group_size}")
        self.in_features = in_features
        self.out_features = out_features
        self.n_groups = in_features // self.cfg.group_size

        self.W_base = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.W_base, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Runtime sidecar state — buffers, never Parameters (I-4).
        self.register_buffer("fast", torch.zeros(out_features, in_features))
        self.register_buffer("trace", torch.zeros(out_features, in_features))
        self.register_buffer("consol", torch.zeros(out_features, in_features))

        # Meta-parameters (trained only in the outer loop).
        self.gate_net = GateNet(self.cfg)
        self.log_plastic_lr = nn.Parameter(
            torch.tensor(math.log(self.cfg.plastic_lr)))

        self.mode = "pretrain"
        self._trace_updates_since_tick = 0

    # ------------------------------------------------------------------ #
    # forward path                                                       #
    # ------------------------------------------------------------------ #

    def effective_weight(self, fast: torch.Tensor | None = None) -> torch.Tensor:
        """W_eff = W_base + alpha * tanh(fast).  tanh bounds the fast
        contribution unconditionally (I-1): even a runaway trace cannot push
        any single weight further than alpha from its base value."""
        f = self.fast if fast is None else fast
        return self.W_base + self.cfg.alpha * torch.tanh(f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.effective_weight(), self.bias)
        if self.mode == "runtime":
            self._accumulate_trace(x, y)
        return y

    @torch.no_grad()
    def _accumulate_trace(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Hebbian eligibility: EMA of outer(post, pre), batch/time-averaged.
        Uses ONLY locally available signals (I-5): the activations this layer
        actually saw and produced. Activations are L2-normalized per feature
        so trace magnitude reflects co-activity pattern, not raw scale."""
        xf = x.detach().reshape(-1, self.in_features)
        yf = y.detach().reshape(-1, self.out_features)
        xf = F.normalize(xf, dim=-1)
        yf = F.normalize(yf, dim=-1)
        outer = yf.t() @ xf / max(xf.shape[0], 1)          # (out, in)
        d = self.cfg.trace_decay
        self.trace.mul_(d).add_(outer, alpha=(1.0 - d))
        self._trace_updates_since_tick += 1

    # ------------------------------------------------------------------ #
    # the slow tick — the only mutation site for fast/consol (I-6)       #
    # ------------------------------------------------------------------ #

    def _group_view(self, t: torch.Tensor) -> torch.Tensor:
        return t.view(self.out_features, self.n_groups, self.cfg.group_size)

    def _group_features(self, neuromod: float,
                        fast: torch.Tensor, trace: torch.Tensor,
                        consol: torch.Tensor) -> torch.Tensor:
        gt = self._group_view(trace)
        gf = self._group_view(fast)
        gc = self._group_view(consol)
        feats = torch.stack([
            gt.abs().mean(-1),
            gt.std(-1, unbiased=False),
            gf.abs().mean(-1),
            gc.mean(-1),
            torch.full_like(gt[..., 0], float(neuromod)),
        ], dim=-1)                                          # (out, n_groups, F)
        return feats

    def _budget_rescale(self, fast: torch.Tensor) -> torch.Tensor:
        """Heterosynaptic normalization (I-2): per group, the L2 norm of the
        fast contribution may not exceed budget_frac * L2 norm of W_base.
        Groups over budget are rescaled down; under-budget groups untouched.
        This is the shared-resource budget from the conversation — when the
        cluster strengthens somewhere it must yield somewhere."""
        contrib = self.cfg.alpha * torch.tanh(fast)
        gnorm = self._group_view(contrib).norm(dim=-1)                  # (out, G)
        bnorm = self._group_view(self.W_base.detach()).norm(dim=-1)
        limit = self.cfg.budget_frac * bnorm + 1e-8
        scale = torch.clamp(limit / (gnorm + 1e-8), max=1.0)            # <=1 only
        return fast * scale.unsqueeze(-1).expand_as(
            self._group_view(fast)).reshape_as(fast)

    @torch.no_grad()
    def tick(self, neuromod: float = 0.0) -> dict:
        """Runtime slow pass. Local signals + one broadcast scalar in;
        bounded, budgeted state change out. No gradients (I-4)."""
        if self.mode != "runtime":
            return {}
        cfg = self.cfg
        feats = self._group_features(neuromod, self.fast, self.trace, self.consol)
        gate = self.gate_net(feats)                          # (out, n_groups)
        gate_w = gate.unsqueeze(-1).expand(
            -1, -1, cfg.group_size).reshape_as(self.fast)

        lr = self.log_plastic_lr.exp().item()
        delta = lr * gate_w * self.trace
        delta = delta / (1.0 + self.consol)                  # consolidation shields

        self.fast.mul_(cfg.fast_decay).add_(delta)
        self.fast.copy_(self._budget_rescale(self.fast))

        # Consolidate where gated updates keep agreeing with accumulated fast.
        agree = (torch.sign(delta) == torch.sign(self.fast)).float()
        self.consol.mul_(cfg.consol_decay).add_(
            cfg.consol_rate * gate_w * delta.abs() * agree)

        self.trace.mul_(0.5)
        self._trace_updates_since_tick = 0
        return {
            "gate_mean": gate.mean().item(),
            "fast_abs_mean": self.fast.abs().mean().item(),
            "consol_mean": self.consol.mean().item(),
        }

    # ------------------------------------------------------------------ #
    # meta-training (differentiable functional inner loop)               #
    # ------------------------------------------------------------------ #

    def functional_forward(self, x: torch.Tensor,
                           fast: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(fast), self.bias)

    def functional_tick(self, fast: torch.Tensor, trace: torch.Tensor,
                        consol: torch.Tensor, neuromod: float
                        ) -> tuple[torch.Tensor, torch.Tensor]:
        """Differentiable mirror of tick() for the outer loop. Operates on
        tensors passed in (never the buffers), so autograd reaches GateNet
        and log_plastic_lr while runtime state stays gradient-free (I-4)."""
        cfg = self.cfg
        feats = self._group_features(neuromod, fast, trace, consol)
        gate = self.gate_net(feats)
        gate_w = gate.unsqueeze(-1).expand(
            -1, -1, cfg.group_size).reshape_as(fast)
        delta = self.log_plastic_lr.exp() * gate_w * trace / (1.0 + consol)
        new_fast = cfg.fast_decay * fast + delta

        contrib = cfg.alpha * torch.tanh(new_fast)
        gnorm = self._group_view(contrib).norm(dim=-1)
        bnorm = self._group_view(self.W_base.detach()).norm(dim=-1)
        limit = cfg.budget_frac * bnorm + 1e-8
        scale = torch.clamp(limit / (gnorm + 1e-8), max=1.0)
        new_fast = new_fast * scale.unsqueeze(-1).expand(
            -1, -1, cfg.group_size).reshape_as(new_fast)
        return new_fast, gate

    # ------------------------------------------------------------------ #
    # mode management                                                    #
    # ------------------------------------------------------------------ #

    def set_mode(self, mode: str) -> None:
        assert mode in ("pretrain", "runtime", "meta")
        self.mode = mode
        self.W_base.requires_grad_(mode == "pretrain")
        if self.bias is not None:
            self.bias.requires_grad_(mode == "pretrain")
        for p in self.gate_net.parameters():
            p.requires_grad_(mode == "meta")
        self.log_plastic_lr.requires_grad_(mode == "meta")

    @torch.no_grad()
    def reset_substrate(self) -> None:
        self.fast.zero_()
        self.trace.zero_()
        self.consol.zero_()
        self._trace_updates_since_tick = 0
