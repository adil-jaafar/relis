import random
import pytest
import torch
from relis.tape import codes as C
from relis.infer import constrain as K


ALL_STATES = (K.S0, K.S1, K.S2, K.S3, K.SA0, K.SED, K.SF0, K.SF4)


def test_allowed_matches_advance_on_every_byte():
    for st in ALL_STATES:
        allowed = K.text_allowed(st)
        for b in range(256):
            ok = True
            try:
                K.advance(st, b)
            except ValueError:
                ok = False
            assert ok == (b in allowed), (st, hex(b))


def test_random_walks_always_decode():
    rng = random.Random(0)
    for _ in range(40):
        g = K.Utf8Guard()
        out = bytearray()
        for _ in range(200):
            out.append(rng.choice(sorted(g.allowed())))
            g.feed(out[-1])
        while not g.at_boundary:                 # terminer le caractère en cours
            out.append(sorted(g.allowed())[0]); g.feed(out[-1])
        bytes(out).decode("utf-8")               # strict : lève si invalide


def test_restricted_ranges_reject_overlong_and_surrogates():
    assert 0x80 not in K.text_allowed(K.SA0) and 0xA0 in K.text_allowed(K.SA0)   # E0
    assert 0xA0 not in K.text_allowed(K.SED) and 0x9F in K.text_allowed(K.SED)   # ED, demi-codets
    assert 0x80 not in K.text_allowed(K.SF0) and 0x90 in K.text_allowed(K.SF0)   # F0
    assert 0x90 not in K.text_allowed(K.SF4) and 0x8F in K.text_allowed(K.SF4)   # F4


def test_illegal_lead_bytes_are_never_allowed():
    s0 = K.text_allowed(K.S0)
    for b in (0xC0, 0xC1, 0xF5, 0xFF, 0x80, 0xBF):
        assert b not in s0
    for b in C.CONTROL_CODES:
        assert b not in s0                       # les codes passent par le masque de phase
    for b in (0x09, 0x0A, 0x0D):
        assert b in s0


def test_guard_boundary_tracking():
    g = K.Utf8Guard()
    assert g.at_boundary
    g.feed(0xC3)
    assert not g.at_boundary
    g.feed(0xA9)                                 # « é »
    assert g.at_boundary


def test_phase_sets_exclude_recall():
    assert C.RECALL not in K.GEN_CODES and C.RECALL not in K.NOTE_CODES
    assert K.AFTER_HDR == frozenset({C.READ, C.SKIP})
    assert K.AFTER_DEC_FULL == frozenset({C.CONT, C.STOP, C.NEXT})
    assert K.GEN_CODES == frozenset({C.END, C.REFRESH, C.NOTE})
    assert K.NOTE_CODES == frozenset({C.REFRESH})


def test_masked_argmax_respects_allowed():
    logits = torch.zeros(256); logits[C.CONT] = 10.0; logits[C.STOP] = 5.0
    assert K.masked_argmax(logits, frozenset({C.STOP, C.NEXT})) == C.STOP
    assert K.masked_argmax(logits, K.AFTER_DEC_FULL) == C.CONT


def test_mask_is_additive_and_finite_where_allowed():
    m = K.mask_for(frozenset({C.END}), "cpu")
    assert m.shape == (256,) and m[C.END].item() == 0.0
    assert torch.isinf(m[C.CONT]) and m[C.CONT].item() < 0
