import random
import torch
from relis.model.config import RelisConfig
from relis.model.relis import RelisModel
from relis.train.checkpoint import save_checkpoint
from relis.data.episodes import Case, generate_case
from relis.infer.controller import Budget, Controller, Event
from relis.eval.harness import judge, load_controller, run_case, summarise


def _ctl():
    return Controller(RelisModel(RelisConfig.tiny()).eval(), "cpu",
                      Budget(max_gen_bytes=24, max_read_bytes=4000))


def test_run_case_reports_counters_and_stop():
    c = generate_case(random.Random(1), "fact_recall")
    r = run_case(_ctl(), c)
    assert set(r) >= {"answer", "read", "written", "saved", "stop_at", "read_segments"}
    assert r["written"] <= 24
    assert r["stop_at"] is None or isinstance(r["stop_at"], int)


def test_judge_absent_and_value():
    c = generate_case(random.Random(2), "absent")
    assert judge(c, "Je ne trouve pas cette information.") is True
    assert judge(c, "C'est 42.") is False
    f = generate_case(random.Random(3), "fact_recall")
    assert judge(f, f"La réponse est {f.answer_value} voilà") is True
    assert judge(f, "aucune idée") is False


def test_load_controller_from_local_checkpoint(tmp_path):
    cfg = RelisConfig.tiny()
    save_checkpoint(str(tmp_path), RelisModel(cfg), None, None, 11, cfg.to_dict(), {})
    ctl, step = load_controller(str(tmp_path / "last.pt"), "cpu")
    assert step == 11 and isinstance(ctl, Controller)


def test_run_case_counters_describe_the_first_pass_only():
    """Un REFRESH relance `scan()` depuis le début (spec §7.1) : les événements d'un
    tour comportent alors les segments de PLUSIEURS passes. `summarise` doit s'arrêter
    à la première décision STOP, pour que `stop_at` et `doc_actions` décrivent
    exactement la même passe que `case.stop_index` — pas un mélange gonflé par les
    passes suivantes."""
    events = [
        Event("turn_start", text="q"),
        # --- passe 1 : historique seul, STOP au 3e segment (index 2) ---
        Event("channel", channel="history", size=3),
        Event("segment", channel="history", header="h0", size=1, action="read"),
        Event("decision", channel="history", text="CONT"),
        Event("segment", channel="history", header="h1", size=1, action="skip"),
        Event("decision", channel="history", text="CONT"),
        Event("segment", channel="history", header="h2", size=1, action="read"),
        Event("decision", channel="history", text="STOP"),
        Event("scan_end", text="stop"),
        Event("gen_byte", byte=65),
        Event("gen_byte", byte=66),
        Event("note", text="suite"),
        Event("refresh", index=1),
        # --- passe 2 (après REFRESH) : plus d'historique, ET des documents cette
        # fois — rien de tout cela ne doit compter dans le résumé de la passe 1 ---
        Event("channel", channel="history", size=2),
        Event("segment", channel="history", header="h0b", size=1, action="read"),
        Event("decision", channel="history", text="CONT"),
        Event("segment", channel="history", header="h1b", size=1, action="read"),
        Event("decision", channel="history", text="NEXT"),
        Event("channel", channel="docs", size=2),
        Event("segment", channel="docs", header="d0", size=1, action="skip"),
        Event("decision", channel="docs", text="CONT"),
        Event("segment", channel="docs", header="d1", size=1, action="read"),
        Event("decision", channel="docs", text="STOP"),
        Event("scan_end", text="stop"),
        Event("gen_byte", byte=67),
        Event("end"),
        Event("turn_end", text="AB", stats={"read": 1, "written": 3, "refresh": 1,
                                            "saved": 0.5, "seconds": 0.01}),
    ]
    s = summarise(events)
    assert s["stop_at"] == 2                # index du STOP dans la PREMIÈRE passe
    assert s["read_segments"] == 2           # h0 et h2 lus, dans la passe 1 seulement
    assert s["doc_actions"] == []            # aucun document ouvert pendant la passe 1
    assert s["skipped_docs"] == 0


def test_judge_requires_word_boundaries():
    c = Case(kind="doc_lookup", query=b"q", history=[], docs=[], doc_relevant=[],
             passes_needed=[(set(), None)], answer_parts=[b"Le montant est de 50 euros."],
             notes=[], answer_value="50")
    assert judge(c, "le total est 1500 euros") is False
    assert judge(c, "le total est 50 euros") is True


def test_judge_absent_unchanged():
    c = generate_case(random.Random(2), "absent")
    assert judge(c, "Je ne trouve pas cette information.") is True
    assert judge(c, "C'est 42.") is False
