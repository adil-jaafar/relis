import torch
from relis.model.config import RelisConfig
from relis.model.relis import RelisModel
from relis.model.state import State
from relis.tape.codes import Mode


def _model():
    cfg = RelisConfig.tiny()   # block 32, chunk 8, window 16
    return cfg, RelisModel(cfg)


def test_forward_shapes_and_tied_head():
    cfg, m = _model()
    x = torch.randint(0, 256, (2, 70))
    st = m.new_state(2, x.device)
    logits, st = m(x, int(Mode.SCAN), st)
    assert logits.shape == (2, 70, 256)
    assert m.head_weight().data_ptr() == m.embed.weight.data_ptr()
    assert st.seen == 70 and st.pending.shape[1] == 70 % cfg.block


def test_causality():
    cfg, m = _model()
    x = torch.randint(0, 256, (1, 50))
    l1, _ = m(x, int(Mode.SCAN), m.new_state(1, x.device))
    x2 = x.clone(); x2[0, 40] = (x2[0, 40] + 1) % 256
    l2, _ = m(x2, int(Mode.SCAN), m.new_state(1, x.device))
    assert torch.allclose(l1[:, :40], l2[:, :40], atol=1e-5)
    assert not torch.allclose(l1[:, 40:], l2[:, 40:])


def test_mode_changes_output():
    cfg, m = _model()
    x = torch.randint(0, 256, (1, 12))
    a, _ = m(x, int(Mode.SCAN), m.new_state(1, x.device))
    b, _ = m(x, int(Mode.GENERATE), m.new_state(1, x.device))
    assert not torch.allclose(a, b)


def test_forward_equals_steps_across_block_boundaries():
    cfg, m = _model()
    L = 2 * cfg.block + 13            # traverse deux frontières d'écriture du Buffer
    x = torch.randint(0, 256, (2, L))
    mode = torch.full((2, L), int(Mode.SCAN))
    mode[:, 40:] = int(Mode.GENERATE)
    l_full, st_full = m(x, mode, m.new_state(2, x.device))
    st = m.new_state(2, x.device)
    outs = []
    for t in range(L):
        l_t, st = m.step(x[:, t], mode[:, t], st)
        outs.append(l_t)
    l_steps = torch.stack(outs, dim=1)
    assert torch.allclose(l_full, l_steps, atol=1e-4), (l_full - l_steps).abs().max()
    assert torch.allclose(st_full.slots, st.slots, atol=1e-4)


def test_forward_split_equals_whole():
    cfg, m = _model()
    x = torch.randint(0, 256, (1, 100))
    l_all, st_all = m(x, int(Mode.SCAN), m.new_state(1, x.device))
    st = m.new_state(1, x.device)
    l1, st = m(x[:, :5], int(Mode.SCAN), st)
    l2, st = m(x[:, 5:61], int(Mode.SCAN), st)
    l3, st = m(x[:, 61:], int(Mode.SCAN), st)
    assert torch.allclose(torch.cat([l1, l2, l3], dim=1), l_all, atol=1e-4)
    assert torch.allclose(st.slots, st_all.slots, atol=1e-4)


def test_slots_change_only_at_block_boundaries():
    cfg, m = _model()
    st = m.new_state(1, "cpu")
    s0 = st.slots.clone()
    _, st = m(torch.randint(0, 256, (1, cfg.block - 1)), int(Mode.SCAN), st)
    assert torch.allclose(st.slots, s0)
    _, st = m(torch.randint(0, 256, (1, 1)), int(Mode.SCAN), st)
    assert not torch.allclose(st.slots, s0)
    assert st.pending.shape[1] == 0


def test_memory_is_constant():
    cfg, m = _model()
    st = m.new_state(1, "cpu")
    _, st = m(torch.randint(0, 256, (1, 3 * cfg.block)), int(Mode.SCAN), st)
    size_a = st.size_bytes()
    _, st = m(torch.randint(0, 256, (1, 40 * cfg.block)), int(Mode.SCAN), st)
    assert st.size_bytes() == size_a


def test_reset_memory_keeps_slots():
    cfg, m = _model()
    st = m.new_state(1, "cpu")
    _, st = m(torch.randint(0, 256, (1, 2 * cfg.block)), int(Mode.ENCODE), st)
    slots = st.slots.clone()
    st.reset_memory()
    assert torch.allclose(st.slots, slots)
    assert st.seen == 0 and st.pending.shape[1] == 0
    assert all(torch.count_nonzero(g[0]) == 0 for g in st.gdn if g is not None)


def test_parameter_count_v1_in_range():
    m = RelisModel(RelisConfig())
    n = sum(p.numel() for p in m.parameters())
    assert 100e6 < n < 170e6, n


def test_slots_parameter_receives_gradient():
    cfg, m = _model()
    x = torch.randint(0, 256, (1, 2 * cfg.block + 3))
    logits, st = m(x, int(Mode.SCAN), m.new_state(1, "cpu"))
    logits.float().sum().backward()
    assert m.slots.grad is not None
    assert m.slots.grad.abs().sum() > 0
    assert m.writer.gate.weight.grad is not None
