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
        y, S, _ = m(x, None, None)
    assert S.dtype == torch.float32
    assert y.shape == x.shape
