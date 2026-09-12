import random
from relis.tape import codes as C
from relis.tape.tape import build_tape, validate_spec
from relis.data.episodes import (KINDS, Case, case_to_spec, generate_case, generate_episode,
                                 iter_cases)


def test_case_history_is_complete_and_spec_is_truncated():
    c = generate_case(random.Random(4), "fact_recall")
    spec = case_to_spec(c)
    assert len(spec.passes[0].history) <= len(c.history)
    assert spec.passes[0].history[-1].after == C.STOP
    assert c.stop_index == len(spec.passes[0].history) - 1


def test_every_kind_produces_a_valid_case_and_spec():
    rng = random.Random(0)
    for kind in KINDS:
        for _ in range(5):
            c = generate_case(rng, kind)
            assert isinstance(c, Case) and c.query and c.history is not None
            assert len(c.passes_needed) == len(c.answer_parts)
            assert len(c.notes) == len(c.answer_parts) - 1
            spec = case_to_spec(c)
            validate_spec(spec)
            assert build_tape(spec).decisions()[-1] == C.END


def test_answer_value_appears_in_the_answer_when_expected():
    rng = random.Random(9)
    for kind in ("fact_recall", "what_did_i_say", "doc_lookup", "variable_tracking"):
        c = generate_case(rng, kind)
        joined = b"".join(c.answer_parts).decode("utf-8", "replace")
        assert c.answer_value and c.answer_value in joined, kind


def test_absent_has_no_answer_value():
    c = generate_case(random.Random(2), "absent")
    assert c.answer_value == ""
    assert c.passes_needed == [(set(), None)]


def test_generate_episode_still_deterministic_and_bounded():
    a = [bytes(build_tape(s).data) for s in (generate_episode(random.Random(7)) for _ in range(20))]
    b = [bytes(build_tape(s).data) for s in (generate_episode(random.Random(7)) for _ in range(20))]
    assert a == b
    assert all(len(t) <= 15_000 for t in a)


def test_iter_cases_is_deterministic():
    x = [c.query for c in iter_cases(3, 15)]
    y = [c.query for c in iter_cases(3, 15)]
    assert x == y
