import torch
from relis.model.config import RelisConfig
from relis.model.relis import RelisModel
from relis.tape.codes import Mode


def _m():
    cfg = RelisConfig.tiny()
    return cfg, RelisModel(cfg).eval()


def test_reset_mask_equals_reset_memory_mid_block():
    cfg, m = _m()
    L, t = 2 * cfg.block + 20, cfg.block + 7
    x = torch.randint(0, 256, (1, L))
    mask = torch.zeros(1, L, dtype=torch.bool); mask[0, t] = True
    la, sa = m(x, int(Mode.SCAN), m.new_state(1, "cpu"), reset=mask)
    st = m.new_state(1, "cpu")
    l1, st = m(x[:, :t], int(Mode.SCAN), st)
    st.reset_memory()
    l2, st = m(x[:, t:], int(Mode.SCAN), st)
    assert torch.allclose(la, torch.cat([l1, l2], 1), atol=1e-4), (la - torch.cat([l1, l2], 1)).abs().max()
    assert torch.allclose(sa.slots, st.slots, atol=1e-4)


def test_reset_per_sample_in_batch_equals_individual_runs():
    cfg, m = _m()
    L = 2 * cfg.block + 9
    x = torch.randint(0, 256, (2, L))
    mask = torch.zeros(2, L, dtype=torch.bool); mask[0, 13] = True; mask[1, cfg.block + 3] = True
    lb, _ = m(x, int(Mode.SCAN), m.new_state(2, "cpu"), reset=mask)
    for b in range(2):
        li, _ = m(x[b:b + 1], int(Mode.SCAN), m.new_state(1, "cpu"), reset=mask[b:b + 1])
        assert torch.allclose(lb[b:b + 1], li, atol=1e-4)


def test_step_reset_equals_forward_reset():
    cfg, m = _m()
    L, t = cfg.block + 11, 6
    x = torch.randint(0, 256, (1, L))
    mask = torch.zeros(1, L, dtype=torch.bool); mask[0, t] = True
    lf, _ = m(x, int(Mode.SCAN), m.new_state(1, "cpu"), reset=mask)
    st = m.new_state(1, "cpu"); outs = []
    for i in range(L):
        lg, st = m.step(x[:, i], int(Mode.SCAN), st, reset=mask[:, i])
        outs.append(lg)
    assert torch.allclose(lf, torch.stack(outs, 1), atol=1e-4)


def test_two_resets_in_one_chunk_and_no_nan():
    cfg, m = _m()
    L = cfg.block
    x = torch.randint(0, 256, (1, L))
    mask = torch.zeros(1, L, dtype=torch.bool); mask[0, 2] = True; mask[0, 4] = True
    lg, st = m(x, int(Mode.SCAN), m.new_state(1, "cpu"), reset=mask)
    assert torch.isfinite(lg).all()
    st2 = m.new_state(1, "cpu")
    _, st2 = m(x[:, :4], int(Mode.SCAN), st2, reset=mask[:, :4]); st2.reset_memory()
    l2, st2 = m(x[:, 4:], int(Mode.SCAN), st2)
    assert torch.allclose(lg[:, 4:], l2, atol=1e-4)


def test_slots_reset_at_block_boundary_equals_fresh_state():
    cfg, m = _m()
    L = 2 * cfg.block
    x = torch.randint(0, 256, (2, L))
    reset = torch.zeros(2, L, dtype=torch.bool); reset[0, cfg.block] = True
    sreset = reset.clone()
    lb, sb = m(x, int(Mode.SCAN), m.new_state(2, "cpu"), reset=reset, slots_reset=sreset)
    # échantillon 0 : après la frontière, tout doit égaler un état neuf sur x[0, block:]
    lf, sf = m(x[0:1, cfg.block:], int(Mode.SCAN), m.new_state(1, "cpu"))
    assert torch.allclose(lb[0:1, cfg.block:], lf, atol=1e-4)
    assert torch.allclose(sb.slots[0:1], sf.slots, atol=1e-4)
    # échantillon 1 : inchangé par rapport à un run sans drapeaux
    ln, sn = m(x[1:2], int(Mode.SCAN), m.new_state(1, "cpu"))
    assert torch.allclose(lb[1:2], ln, atol=1e-4)
    assert torch.allclose(sb.slots[1:2], sn.slots, atol=1e-4)


def test_slots_reset_off_boundary_is_rejected():
    cfg, m = _m()
    x = torch.randint(0, 256, (1, cfg.block))
    sreset = torch.zeros(1, cfg.block, dtype=torch.bool); sreset[0, 5] = True
    try:
        m(x, int(Mode.SCAN), m.new_state(1, "cpu"), slots_reset=sreset)
        assert False, "attendu : AssertionError"
    except AssertionError:
        pass


def test_reset_memory_new_semantics():
    cfg, m = _m()
    st = m.new_state(1, "cpu")
    _, st = m(torch.randint(0, 256, (1, cfg.block + 5)), int(Mode.SCAN), st)
    pend, seen = st.pending.clone(), st.seen
    conv_before = [g[1].clone() for g in st.gdn if g is not None]
    st.reset_memory()
    assert torch.equal(st.pending, pend) and st.seen == seen
    assert all(torch.count_nonzero(g[0]) == 0 for g in st.gdn if g is not None)
    assert all(torch.equal(g[1], c) for g, c in zip([g for g in st.gdn if g is not None], conv_before))
    assert all(c["k"].shape[2] == 0 and c["kseg"].shape[1] == 0 for c in st.swa if c is not None)
    st.reset_all()
    assert st.pending.shape[1] == 0 and st.seen == 0


def test_grad_flows_with_masks():
    cfg, m = _m(); m.train()
    L = 2 * cfg.block + 3
    x = torch.randint(0, 256, (2, L))
    reset = torch.zeros(2, L, dtype=torch.bool); reset[0, 10] = True; reset[1, cfg.block] = True
    sreset = torch.zeros(2, L, dtype=torch.bool); sreset[1, cfg.block] = True
    lg, _ = m(x, int(Mode.SCAN), m.new_state(2, "cpu"), reset=reset, slots_reset=sreset)
    lg.float().sum().backward()
    assert m.slots.grad is not None and torch.isfinite(m.slots.grad).all()
