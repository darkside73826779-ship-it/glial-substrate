"""TinyTransformer with substrate FFN layers, and the SubstrateController.

Design choices (see DESIGN.md):
  - Substrate lives in the FFN only. FFN layers are the transformer's
    key-value associative memory (the solid line in the cross-map), so that
    is where runtime-writable memory belongs. Attention stays vanilla.
  - The controller owns the global neuromodulator: a single broadcast scalar
    derived from prediction surprise (fast-EMA loss vs slow-EMA loss).
    High recent surprise -> plasticity permitted. This is the dopamine/ACh
    analog: one number, broadcast to every glial gate (I-5).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import SubstrateConfig, SubstrateLinear


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 cfg: SubstrateConfig, use_substrate: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        lin = (lambda i, o: SubstrateLinear(i, o, cfg)) if use_substrate \
            else (lambda i, o: nn.Linear(i, o))
        self.ff_in = lin(d_model, d_ff)
        self.ff_out = lin(d_ff, d_model)

    def forward(self, x: torch.Tensor, fast_map: dict | None = None,
                act_out: dict | None = None) -> torch.Tensor:
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool),
                          diagonal=1)
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        h2 = self.ln2(x)
        # Sanctioned first-class functional path (RULINGS D1). When fast_map is
        # None the code path is bit-identical to the vanilla forward (I-7, I-11).
        if fast_map is None or not isinstance(self.ff_in, SubstrateLinear):
            x = x + self.ff_out(F.gelu(self.ff_in(h2)))
        else:
            hin = self.ff_in.functional_forward(h2, fast_map[self.ff_in])
            if act_out is not None:
                act_out[self.ff_in] = (h2, hin)
            g = F.gelu(hin)
            hout = self.ff_out.functional_forward(g, fast_map[self.ff_out])
            if act_out is not None:
                act_out[self.ff_out] = (g, hout)
            x = x + hout
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 4, d_ff: int = 256, max_len: int = 256,
                 cfg: SubstrateConfig | None = None,
                 use_substrate: bool = True):
        super().__init__()
        self.cfg = cfg or SubstrateConfig()
        self.use_substrate = use_substrate
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff, self.cfg, use_substrate)
            for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight  # tied

    def forward(self, idx: torch.Tensor, fast_map: dict | None = None,
                act_out: dict | None = None) -> torch.Tensor:
        """Standard forward when fast_map is None. When fast_map (a dict mapping
        each SubstrateLinear to an episode-local fast tensor) is provided, the
        FFN substrate layers route through functional_forward with those
        tensors, threading autograd to GateNet for meta-training (RULINGS D1).
        Optional act_out dict receives per-layer (input, output) activations for
        the differentiable trace mirror. Runtime buffers are never touched."""
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        for b in self.blocks:
            x = b(x, fast_map=fast_map, act_out=act_out)
        return self.head(self.ln_f(x))

    def substrate_layers(self) -> list[SubstrateLinear]:
        return [m for m in self.modules() if isinstance(m, SubstrateLinear)]

    def set_mode(self, mode: str) -> None:
        for l in self.substrate_layers():
            l.set_mode(mode)
        # Non-substrate params follow pretrain vs frozen.
        pretrain = (mode == "pretrain")
        for n, p in self.named_parameters():
            if "gate_net" in n or "log_plastic_lr" in n or "W_base" in n:
                continue
            if n.endswith(".bias") and any(
                    isinstance(self.get_submodule(n.rsplit(".", 1)[0]),
                               SubstrateLinear) for _ in [0]):
                continue
            p.requires_grad_(pretrain)

    def reset_substrate(self) -> None:
        for l in self.substrate_layers():
            l.reset_substrate()


class SubstrateController:
    """Owns tick scheduling and the global neuromodulator.

    neuromod = sigmoid(k * (slow_ema_loss - fast_ema_loss) / slow_ema_loss)
    centered so that when recent loss is *improving or surprising* relative
    to baseline, the scalar rises above its resting value.
    Resting value ~0.5 with k=0 drift; gates learn to interpret it.
    """

    def __init__(self, model: TinyTransformer, tick_every: int = 8,
                 k: float = 8.0):
        self.model = model
        self.tick_every = tick_every
        self.k = k
        self.step = 0
        self.fast_ema: float | None = None
        self.slow_ema: float | None = None
        self.last_stats: list[dict] = []

    def observe_loss(self, loss: float) -> None:
        self.fast_ema = loss if self.fast_ema is None else \
            0.7 * self.fast_ema + 0.3 * loss
        self.slow_ema = loss if self.slow_ema is None else \
            0.99 * self.slow_ema + 0.01 * loss

    def neuromod(self) -> float:
        if self.fast_ema is None or self.slow_ema is None:
            return 0.5
        rel = (self.slow_ema - self.fast_ema) / (abs(self.slow_ema) + 1e-8)
        return float(torch.sigmoid(torch.tensor(self.k * rel)))

    def maybe_tick(self) -> list[dict]:
        self.step += 1
        if self.step % self.tick_every != 0:
            return []
        m = self.neuromod()
        self.last_stats = [l.tick(m) for l in self.model.substrate_layers()]
        return self.last_stats
