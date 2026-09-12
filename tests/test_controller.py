import torch
from relis.model.config import RelisConfig
from relis.model.relis import RelisModel
from relis.tape import codes as C
from relis.tape.tape import Segment
from relis.tape.headers import format_header
from relis.infer.controller import Budget, Controller, Event


def _seg(role, txt, i=0):
    return Segment(header=format_header({"role": role, "i": i}), content=txt.encode("utf-8"))


def _ctl(**kw):
    m = RelisModel(RelisConfig.tiny()).eval()
    return Controller(m, "cpu", Budget(**kw))


def _kinds(events):
    return [e.kind for e in events]


def test_turn_without_history_reaches_generation():
    ev = list(_ctl(max_gen_bytes=12).turn(b"bonjour"))
    assert _kinds(ev)[0] == "turn_start" and _kinds(ev)[-1] == "turn_end"
    assert "channel" not in _kinds(ev)          # aucun canal ouvert
    assert ev[-1].stats["written"] <= 12


def test_scan_emits_one_segment_event_per_visited_segment():
    hist = [_seg("user", f"tour {i}", i) for i in range(4)]
    ev = list(_ctl(max_gen_bytes=8).turn(b"q", history=hist))
    segs = [e for e in ev if e.kind == "segment"]
    assert 1 <= len(segs) <= 4
    assert all(e.action in ("read", "skip") for e in segs)
    assert all(e.header for e in segs)


def test_last_history_segment_cannot_continue_without_docs():
    hist = [_seg("user", "a", 0)]
    ev = list(_ctl(max_gen_bytes=4).turn(b"q", history=hist))
    dec = [e.text for e in ev if e.kind == "decision"]
    assert dec == ["STOP"]                      # CONT et NEXT sont masqués


def test_last_history_segment_may_go_next_when_docs_exist():
    hist = [_seg("user", "a", 0)]
    docs = [Segment(header=format_header({"type": "doc", "name": "d.txt"}), content=b"x")]
    ev = list(_ctl(max_gen_bytes=4).turn(b"q", history=hist, docs=docs))
    dec = [e.text for e in ev if e.kind == "decision"]
    assert dec[0] in ("STOP", "NEXT")
    if dec[0] == "NEXT":
        assert [e.channel for e in ev if e.kind == "channel"] == ["history", "docs"]
        assert dec[-1] == "STOP"                # dernier document : seul STOP reste


def test_read_budget_forces_skip_then_stop():
    hist = [_seg("user", "x" * 400, i) for i in range(6)]
    ev = list(_ctl(max_read_bytes=10, max_gen_bytes=4).turn(b"q", history=hist))
    segs = [e for e in ev if e.kind == "segment"]
    assert segs[-1].action == "skip"
    assert [e for e in ev if e.kind == "scan_end"][0].text == "budget"


def test_generated_answer_is_valid_utf8_and_within_budget():
    ev = list(_ctl(max_gen_bytes=64).turn(b"bonjour"))
    raw = bytes(e.byte for e in ev if e.kind == "gen_byte")
    raw.decode("utf-8")                          # strict
    assert len(raw) <= 64
    assert ev[-1].stats["written"] == len(raw)


def test_stats_are_coherent():
    hist = [_seg("user", "abcdefghij", i) for i in range(3)]
    ev = list(_ctl(max_gen_bytes=8).turn(b"q", history=hist))
    st = ev[-1].stats
    assert st["available"] == 30
    assert 0 <= st["read"] <= 30 and 0.0 <= st["saved"] <= 1.0
    assert st["refresh"] <= 8 and st["seconds"] >= 0


def test_refresh_budget_is_respected(monkeypatch):
    """Un modèle truqué qui veut toujours REFRESH doit être arrêté par le budget."""
    ctl = _ctl(max_refresh=2, max_gen_bytes=200)
    real = ctl._decide

    def greedy(logits, allowed):
        if C.REFRESH in allowed:
            return C.REFRESH
        if C.END in allowed:
            return C.END
        return real(logits, allowed)

    ctl._decide = greedy
    ev = list(ctl.turn(b"q", history=[_seg("user", "a", 0)]))
    assert len([e for e in ev if e.kind == "refresh"]) == 2
    assert ev[-1].stats["refresh"] == 2


def test_refresh_resets_memory_but_keeps_slots():
    ctl = _ctl(max_refresh=1, max_gen_bytes=40)
    real = ctl._decide
    fired = {"n": 0}

    def once(logits, allowed):
        if C.REFRESH in allowed and fired["n"] == 0:
            fired["n"] = 1
            return C.REFRESH
        return real(logits, allowed)

    ctl._decide = once
    slots_before = []
    ev = []
    for e in ctl.turn(b"q", history=[_seg("user", "abc", 0)]):
        if e.kind == "refresh":
            slots_before.append(ctl.state.slots.clone())
            assert all(torch.count_nonzero(g[0]) == 0 for g in ctl.state.gdn if g is not None)
        ev.append(e)
    assert fired["n"] == 1 and slots_before
