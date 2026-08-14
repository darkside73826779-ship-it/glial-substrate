"""Mechanical checks of the DESIGN.md invariants.

These are not unit tests of convenience — they are the contract. Any edit
to substrate/ must leave every one of these passing. An agent that
"cleans up" the code and breaks one of these has broken the design.

Run:  python scripts/test_invariants.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from substrate import (SubstrateConfig, SubstrateController, SubstrateLinear,
                       TinyTransformer)

PASS, FAIL = "PASS", "FAIL"
failures = []


def check(name, cond, detail=""):
    print(f"[{PASS if cond else FAIL}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


def main():
    torch.manual_seed(0)
    cfg = SubstrateConfig(group_size=4, alpha=0.1, budget_frac=0.25)
    lay = SubstrateLinear(16, 8, cfg)

    # I-1: fast contribution is hard-bounded by alpha via tanh.
    with torch.no_grad():
        lay.fast.fill_(1e9)
        dev = (lay.effective_weight() - lay.W_base).abs().max().item()
    check("I-1 bounded fast contribution", dev <= cfg.alpha + 1e-6,
          f"max dev {dev:.4f} <= alpha {cfg.alpha}")
    lay.reset_substrate()

    # I-2: budget holds after tick even under adversarial traces.
    lay.set_mode("runtime")
    with torch.no_grad():
        lay.trace.fill_(100.0)
        lay.log_plastic_lr.fill_(5.0)  # absurd lr on purpose
    for _ in range(20):
        lay.tick(neuromod=1.0)
    with torch.no_grad():
        contrib = cfg.alpha * torch.tanh(lay.fast)
        gnorm = contrib.view(8, 4, 4).norm(dim=-1)
        bnorm = lay.W_base.view(8, 4, 4).norm(dim=-1)
        ok = bool((gnorm <= cfg.budget_frac * bnorm + 1e-4).all())
    check("I-2 per-group budget enforced", ok)

    # I-3: grouping is along fan-in (functional neighborhood).
    check("I-3 fan-in grouping", lay.n_groups == 16 // cfg.group_size)

    # I-4a: runtime tick creates no autograd graph and no grads.
    lay2 = SubstrateLinear(16, 8, cfg)
    lay2.set_mode("runtime")
    x = torch.randn(4, 16)
    y = lay2(x)
    lay2.tick(0.5)
    check("I-4a runtime forward detached from tick state",
          not lay2.fast.requires_grad and not lay2.trace.requires_grad)
    grads = [p.grad for p in lay2.gate_net.parameters()]
    check("I-4b no gradients into gate_net at runtime",
          all(g is None for g in grads))

    # I-4c: base weights frozen at runtime.
    check("I-4c W_base frozen at runtime", not lay2.W_base.requires_grad)

    # I-4d: meta mode DOES reach gate_net through functional_tick.
    lay3 = SubstrateLinear(16, 8, cfg)
    lay3.set_mode("meta")
    fast = torch.zeros(8, 16)
    trace = torch.randn(8, 16)
    consol = torch.zeros(8, 16)
    new_fast, gate = lay3.functional_tick(fast, trace, consol, 0.5)
    out = lay3.functional_forward(torch.randn(4, 16), new_fast)
    out.sum().backward()
    got = any(p.grad is not None and p.grad.abs().sum() > 0
              for p in lay3.gate_net.parameters())
    check("I-4d meta gradients reach gate_net", got)

    # I-5: trace uses only local activity (no targets/labels touch it).
    lay4 = SubstrateLinear(16, 8, cfg)
    lay4.set_mode("runtime")
    before = lay4.trace.clone()
    _ = lay4(torch.randn(4, 16))
    check("I-5 trace accumulates from forward alone",
          not torch.equal(before, lay4.trace))

    # I-6: fast/consol mutate ONLY at tick, never during forward.
    lay5 = SubstrateLinear(16, 8, cfg)
    lay5.set_mode("runtime")
    f0, c0 = lay5.fast.clone(), lay5.consol.clone()
    for _ in range(10):
        _ = lay5(torch.randn(4, 16))
    check("I-6 forward never mutates fast/consol",
          torch.equal(f0, lay5.fast) and torch.equal(c0, lay5.consol))

    # I-7: pretrain mode leaves substrate inert.
    lay6 = SubstrateLinear(16, 8, cfg)
    lay6.set_mode("pretrain")
    _ = lay6(torch.randn(4, 16))
    lay6.tick(1.0)
    check("I-7 substrate inert during pretrain",
          lay6.fast.abs().sum() == 0 and lay6.trace.abs().sum() == 0)

    # I-8: stability under sustained runtime load (no NaN/explosion).
    model = TinyTransformer(32, d_model=32, n_heads=2, n_layers=2, d_ff=64,
                            cfg=cfg)
    model.set_mode("runtime")
    ctrl = SubstrateController(model, tick_every=2)
    with torch.no_grad():
        for _ in range(60):
            idx = torch.randint(32, (4, 16))
            out = model(idx)
            ctrl.observe_loss(float(out.float().var()))
            ctrl.maybe_tick()
    finite = all(torch.isfinite(l.fast).all() and torch.isfinite(l.consol).all()
                 for l in model.substrate_layers())
    check("I-8 stable under sustained ticking", finite and
          torch.isfinite(out).all())

    # I-9: baseline/substrate parameter parity of the forward path.
    torch.manual_seed(1)
    m_sub = TinyTransformer(32, d_model=32, n_heads=2, n_layers=2, d_ff=64,
                            cfg=cfg, use_substrate=True)
    torch.manual_seed(1)
    m_base = TinyTransformer(32, d_model=32, n_heads=2, n_layers=2, d_ff=64,
                             cfg=cfg, use_substrate=False)
    n_sub = sum(p.numel() for n, p in m_sub.named_parameters()
                if "gate_net" not in n and "log_plastic_lr" not in n)
    n_base = sum(p.numel() for p in m_base.parameters())
    check("I-9 forward-path parameter parity", n_sub == n_base,
          f"{n_sub} vs {n_base}")

    # I-11: first-class functional forward (RULINGS D1) must be bit-identical to
    # runtime-mode forward when the supplied fast tensors equal the module
    # buffers. This is the guarantee that the sanctioned meta path did not
    # perturb deployment behavior.
    torch.manual_seed(3)
    m11 = TinyTransformer(32, d_model=32, n_heads=2, n_layers=2, d_ff=64,
                          cfg=cfg)
    m11.set_mode("runtime")
    idx = torch.randint(32, (4, 12))
    with torch.no_grad():
        for l in m11.substrate_layers():
            l.fast.normal_(0, 0.1)  # nontrivial fast so the check has teeth
        y_runtime = m11(idx)
        fast_map = {l: l.fast.clone() for l in m11.substrate_layers()}
        y_func = m11(idx, fast_map=fast_map)
    dev11 = (y_runtime - y_func).abs().max().item()
    check("I-11 functional forward matches runtime forward",
          torch.allclose(y_runtime, y_func, atol=1e-6), f"max|d| {dev11:.2e}")

    # ---- DG module invariants (I-12..I-16 = I-DG-1..4 + pin-1 linearity) ---- #
    from substrate.dg import (DGConfig, DGProjection, SparseAssociativeMemory)
    dcfg = DGConfig(d_in=256, d_dg=1024, k=16, mode="random")
    dgp = DGProjection(dcfg)
    sam = SparseAssociativeMemory(d_value=32, cfg=dcfg)

    # I-12 (I-DG-1 firewall): random-mode P and M are buffers, never Parameters.
    pnames = [n for n, _ in dgp.named_parameters()]
    mnames = [n for n, _ in sam.named_parameters()]
    check("I-12 DG firewall: random P + M are buffers",
          "P" not in pnames and not isinstance(sam.M, torch.nn.Parameter)
          and mnames == [])

    # I-13 (I-DG-2 shared space): one deterministic encode path (write == read).
    hk = torch.randn(4, 256)
    dgp.adapt(hk)
    check("I-13 DG shared-space encode deterministic",
          torch.equal(dgp.encode(hk), dgp.encode(hk)))

    # I-14 (I-DG-3 locality): encode depends on its input only, forward-only
    # (random mode: no gradient path into P or the code).
    code = dgp.encode(hk)
    check("I-14 DG encode local + forward-only",
          not code.requires_grad and not dgp.P.requires_grad)

    # I-15 (I-DG-4 bounds): per-column M budget holds after adversarial writes.
    for _ in range(30):
        v = torch.randn(8, 32)
        c = dgp.encode(torch.randn(8, 256))
        sam.write(v, c, gain=5.0)
    coln = sam.M.norm(dim=0)
    check("I-15 DG per-column budget enforced",
          bool((coln <= dcfg.beta + 1e-4).all()))

    # I-16 (pin-1 dead-path ban): the associative RETRIEVAL channel is linear
    # end-to-end (no nonlinearity between M.s and the logit). Superposition of
    # read() must hold; the encode may be nonlinear, the read may not.
    c1 = dgp.encode(torch.randn(3, 256))
    c2 = dgp.encode(torch.randn(3, 256))
    lhs = sam.read(0.3 * c1 + 0.7 * c2)
    rhs = 0.3 * sam.read(c1) + 0.7 * sam.read(c2)
    check("I-16 DG retrieval channel linear (dead-path ban)",
          torch.allclose(lhs, rhs, atol=1e-5))

    print()
    if failures:
        print(f"{len(failures)} INVARIANT(S) BROKEN: {failures}")
        sys.exit(1)
    print("all invariants hold")


if __name__ == "__main__":
    main()
