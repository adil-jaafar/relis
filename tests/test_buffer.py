import torch
from relis.model.config import RelisConfig
from relis.model.buffer import SlotRead, SlotWrite, init_slots, expand_slots, _mha


def test_slot_read_shape_and_dependence():
    cfg = RelisConfig.tiny()
    r = SlotRead(cfg)
    x = torch.randn(2, 9, cfg.d_model)
    slots = expand_slots(init_slots(cfg), 2)
    y = r(x, slots)
    assert y.shape == x.shape
    slots2 = slots + 1.0
    assert not torch.allclose(y, r(x, slots2))


def test_slot_write_gate_and_shape():
    cfg = RelisConfig.tiny()
    w = SlotWrite(cfg)
    slots = expand_slots(init_slots(cfg), 3)
    h = torch.randn(3, cfg.chunk, cfg.d_model)
    new = w(h, slots)
    assert new.shape == slots.shape
    assert not torch.allclose(new, slots)


def test_apply_blocks_is_sequential_over_chunks():
    cfg = RelisConfig.tiny()   # chunk 8
    w = SlotWrite(cfg)
    slots = expand_slots(init_slots(cfg), 1)
    h = torch.randn(1, 24, cfg.d_model)
    seq = slots
    for c in range(3):
        seq = w(h[:, c * 8:(c + 1) * 8], seq)
    assert torch.allclose(w.apply_blocks(h, slots), seq, atol=1e-6)


def test_init_slots_is_parameter():
    cfg = RelisConfig.tiny()
    p = init_slots(cfg)
    assert isinstance(p, torch.nn.Parameter) and p.shape == (cfg.n_slots, cfg.d_model)
    assert expand_slots(p, 5).shape == (5, cfg.n_slots, cfg.d_model)


def test_mha_fp32_under_autocast():
    """_mha should maintain fp32 inside autocast block."""
    q = torch.randn(1, 5, 32, dtype=torch.float32)
    k = torch.randn(1, 4, 32, dtype=torch.float32)
    v = torch.randn(1, 4, 32, dtype=torch.float32)
    n_heads = 2

    # Inside autocast(bfloat16), _mha should still return fp32
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out_inside = _mha(q, k, v, n_heads)

    # Outside autocast, _mha should return fp32
    out_outside = _mha(q, k, v, n_heads)

    # Both should be fp32
    assert out_inside.dtype == torch.float32
    assert out_outside.dtype == torch.float32

    # Results should match closely
    assert torch.allclose(out_inside, out_outside, atol=1e-5)
