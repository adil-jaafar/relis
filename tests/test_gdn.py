import torch
from relis.model.config import RelisConfig
from relis.model.gdn import gdn_chunked, gdn_step, GatedDeltaNet


def _inputs(B=2, H=2, L=37, dk=8, dv=8):
    q = torch.randn(B, H, L, dk)
    k = torch.nn.functional.normalize(torch.randn(B, H, L, dk), dim=-1)
    v = torch.randn(B, H, L, dv)
    log_alpha = -torch.rand(B, H, L) * 0.5          # α ∈ (0.6, 1)
    beta = torch.sigmoid(torch.randn(B, H, L))
    S = torch.randn(B, H, dk, dv) * 0.1
    return q, k, v, log_alpha, beta, S


def _reference(q, k, v, log_alpha, beta, S):
    outs = []
    for t in range(q.shape[2]):
        o, S = gdn_step(q[:, :, t], k[:, :, t], v[:, :, t], log_alpha[:, :, t], beta[:, :, t], S)
        outs.append(o)
    return torch.stack(outs, dim=2), S


def test_step_matches_definition():
    q, k, v, la, be, S = _inputs(L=1)
    a = la[:, :, 0].exp()[..., None, None]; b = be[:, :, 0][..., None, None]
    kt = k[:, :, 0]; vt = v[:, :, 0]; qt = q[:, :, 0]
    kS = kt.unsqueeze(-2) @ S
    S_ref = a * S + b * (kt.unsqueeze(-1) @ (vt.unsqueeze(-2) - a * kS))
    o_ref = (qt.unsqueeze(-2) @ S_ref).squeeze(-2)
    o, S_new = gdn_step(qt, kt, vt, la[:, :, 0], be[:, :, 0], S)
    assert torch.allclose(S_new, S_ref, atol=1e-6)
    assert torch.allclose(o, o_ref, atol=1e-6)


def test_chunked_equals_recurrent_with_padding():
    q, k, v, la, be, S = _inputs(L=37)           # 37 n'est pas un multiple de 8
    o_ref, S_ref = _reference(q, k, v, la, be, S)
    o, S_new = gdn_chunked(q, k, v, la, be, S, chunk=8)
    assert torch.allclose(o, o_ref, atol=1e-4), (o - o_ref).abs().max()
    assert torch.allclose(S_new, S_ref, atol=1e-4)


def test_chunked_state_carry_across_calls():
    q, k, v, la, be, S = _inputs(L=40)
    o_ref, S_ref = _reference(q, k, v, la, be, S)
    o1, S1 = gdn_chunked(q[:, :, :13], k[:, :, :13], v[:, :, :13], la[:, :, :13], be[:, :, :13], S, chunk=8)
    o2, S2 = gdn_chunked(q[:, :, 13:], k[:, :, 13:], v[:, :, 13:], la[:, :, 13:], be[:, :, 13:], S1, chunk=8)
    assert torch.allclose(torch.cat([o1, o2], dim=2), o_ref, atol=1e-4)
    assert torch.allclose(S2, S_ref, atol=1e-4)


def test_module_forward_equals_steps():
    cfg = RelisConfig.tiny()
    m = GatedDeltaNet(cfg)
    x = torch.randn(2, 21, cfg.d_model)
    y_full, S_full, cs_full = m(x, None, None)
    S, cs = m.init_state(2, x.device)
    ys = []
    for t in range(21):
        y_t, S, cs = m.step(x[:, t:t + 1], S, cs)
        ys.append(y_t)
    y_steps = torch.cat(ys, dim=1)
    assert torch.allclose(y_full, y_steps, atol=1e-4), (y_full - y_steps).abs().max()
    assert torch.allclose(S_full, S, atol=1e-4)
    assert torch.allclose(cs_full, cs, atol=1e-5)


def test_state_is_fp32_under_fp16_inputs():
    cfg = RelisConfig.tiny()
    m = GatedDeltaNet(cfg)
    x = torch.randn(1, 9, cfg.d_model)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y_ac, S_ac, _ = m(x, None, None)
    y_pl, S_pl, _ = m(x, None, None)
    assert S_ac.dtype == torch.float32
    assert y_ac.shape == x.shape
    # The module's qkv/beta/dt linear projections legitimately run in bf16
    # under autocast (only the GDN recurrence core is forced fp32), so their
    # rounding perturbs q/k/v/log_alpha/beta slightly before they even reach
    # gdn_chunked; that perturbation then propagates through 9 recurrent
    # steps. Measured empirically (see task-4-report.md): this yields an S
    # difference of ~2.8e-3 with the fp32-forcing fix applied, and an
    # almost identical ~2.7e-3 without it -- i.e. this end-to-end tolerance
    # does not discriminate the fix and must stay loose. atol=1e-2 keeps it
    # meaningful (an order of magnitude under the y tolerance) without being
    # sensitive to legitimate upstream bf16 rounding.
    assert torch.allclose(S_ac, S_pl, atol=1e-2)
    assert torch.allclose(y_ac.float(), y_pl.float(), atol=5e-2)

    # Load-bearing guard: gdn_step/gdn_chunked themselves must ignore an
    # ambient autocast context entirely when given IDENTICAL fp32 inputs --
    # this is what actually fails without the internal
    # `torch.autocast(device_type=..., enabled=False)` wrapper (measured:
    # ~3.9e-3 / ~2.1e-2 max abs diff on S / o without the wrapper, for the
    # same seed and shapes below), even though the module-level test above
    # cannot discriminate it since it never feeds identical inputs to both
    # branches.
    q, k, v, la, be, S0 = _inputs(B=1, H=cfg.n_heads, L=9, dk=cfg.head_dim, dv=cfg.head_dim)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        o_ac, S_new_ac = gdn_chunked(q, k, v, la, be, S0, chunk=cfg.chunk)
    o_pl, S_new_pl = gdn_chunked(q, k, v, la, be, S0, chunk=cfg.chunk)
    assert o_ac.dtype == torch.float32 and S_new_ac.dtype == torch.float32
    assert torch.allclose(o_ac, o_pl, atol=1e-5)
    assert torch.allclose(S_new_ac, S_new_pl, atol=1e-5)

    qt, kt, vt, lat, bet = q[:, :, 0], k[:, :, 0], v[:, :, 0], la[:, :, 0], be[:, :, 0]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        o_ac2, S_ac2 = gdn_step(qt, kt, vt, lat, bet, S0)
    o_pl2, S_pl2 = gdn_step(qt, kt, vt, lat, bet, S0)
    assert o_ac2.dtype == torch.float32 and S_ac2.dtype == torch.float32
    assert torch.allclose(o_ac2, o_pl2, atol=1e-5)
    assert torch.allclose(S_ac2, S_pl2, atol=1e-5)
