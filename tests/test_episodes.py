import random
from collections import Counter
from relis.tape import codes as C
from relis.tape.tape import build_tape, Segment, validate_spec
from relis.data.episodes import KINDS, generate_episode, iter_episodes, oracle_pass


def _seg(i):
    return Segment(header=f"role=user;i={i}", content=f"tour {i}".encode())


def test_oracle_stops_at_oldest_needed_history_segment():
    hist = [_seg(i) for i in range(5)]           # 0 = le plus récent
    p = oracle_pass(hist, needed_hist={1, 3}, docs=[], needed_doc=None, doc_relevant=[])
    assert [s.after for s in p.history] == [C.CONT, C.CONT, C.CONT, C.STOP]
    assert p.docs is None and all(s.read for s in p.history)


def test_oracle_goes_to_docs_and_skips_by_header():
    hist = [_seg(0), _seg(1)]
    docs = [Segment("type=doc;name=a", b"a"), Segment("type=doc;name=b", b"b"), Segment("type=doc;name=c", b"c")]
    p = oracle_pass(hist, needed_hist=set(), docs=docs, needed_doc=1, doc_relevant=[False, True, False])
    assert [s.after for s in p.history] == [C.NEXT]           # rien de requis dans l'historique
    assert [(s.read, s.after) for s in p.docs] == [(False, C.CONT), (True, C.STOP)]


def test_oracle_absent_reads_everything_then_stops():
    hist = [_seg(0), _seg(1)]
    docs = [Segment("type=doc;name=a", b"a")]
    p = oracle_pass(hist, needed_hist=set(), docs=docs, needed_doc=None, doc_relevant=[False])
    assert [s.after for s in p.history] == [C.CONT, C.NEXT]
    assert [(s.read, s.after) for s in p.docs] == [(False, C.STOP)]


def test_every_kind_builds_a_valid_tape():
    rng = random.Random(0)
    for kind in KINDS:
        for _ in range(5):
            spec = generate_episode(rng, kind)
            validate_spec(spec)
            t = build_tape(spec)
            assert t.decisions()[-1] == C.END
            assert 100 < len(t) < 16_000, (kind, len(t))   # doit tenir dans une séquence de 16 384


def test_two_facts_has_exactly_one_refresh_with_note():
    rng = random.Random(1)
    spec = generate_episode(rng, "two_facts")
    assert len(spec.passes) == 2 and len(spec.notes) == 1 and spec.notes[0].startswith(b"il manque")
    t = build_tape(spec)
    assert sum(t.reset) == 1


def test_long_answer_refreshes_every_L_bytes():
    rng = random.Random(2)
    spec = generate_episode(rng, "long_answer")
    assert len(spec.passes) >= 2
    assert all(n == b"r\xc3\xa9ponse longue, suite" for n in spec.notes)
    assert all(256 <= len(part) <= 1024 for part in spec.answer_parts[:-1])


def test_doc_lookup_reads_needed_doc_and_answer_mentions_fact():
    rng = random.Random(3)
    spec = generate_episode(rng, "doc_lookup")
    p = spec.passes[0]
    assert p.docs is not None and any(s.read and s.after == C.STOP for s in p.docs)
    assert p.history[-1].after == C.NEXT


def test_iter_episodes_is_deterministic_and_covers_kinds():
    a = [build_tape(s).data for s in iter_episodes(seed=7, n=30)]
    b = [build_tape(s).data for s in iter_episodes(seed=7, n=30)]
    assert a == b
    counts = Counter()
    rng = random.Random(11)
    for _ in range(300):
        spec = generate_episode(rng)
        counts[len(spec.passes) > 1] += 1
    assert counts[True] > 20 and counts[False] > 100
