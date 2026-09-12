import random
import torch
from relis.model.config import RelisConfig
from relis.model.relis import RelisModel
from relis.train.checkpoint import save_checkpoint
from relis.tape.headers import format_header
from relis.tape.tape import Segment
from relis.data.episodes import Case, generate_case, iter_cases
from relis.infer.controller import Budget, Controller, Event
from relis.eval.harness import budget_for, judge, load_controller, run_case, summarise
from relis.eval.autonomy import classify, evaluate, format_report
from relis.eval.needle import build_case, evaluate as needle_eval, format_table
from relis.eval.adaptive import evaluate as adaptive_eval


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


def test_classify_covers_every_case():
    # Un document utile sauté doit être détecté via `doc_actions`, pas via un simple
    # compteur : un compteur ne dit pas SI le document nécessaire fait partie des
    # sautés (voir la décision consignée dans le rapport de la tâche 5).
    c = generate_case(random.Random(5), "fact_recall")
    c.stop_index = 3
    assert classify(c, {"stop_at": 3, "doc_actions": []}) == "exact"
    assert classify(c, {"stop_at": 1, "doc_actions": []}) == "trop_tot"
    assert classify(c, {"stop_at": 6, "doc_actions": []}) == "trop_tard"
    assert classify(c, {"stop_at": None, "doc_actions": []}) == "jamais_stop"
    d = generate_case(random.Random(6), "doc_lookup")
    needed_doc = d.passes_needed[0][1]
    doc_actions = ["read"] * (needed_doc + 1)
    doc_actions[needed_doc] = "skip"
    assert classify(d, {"stop_at": -1, "doc_actions": doc_actions}) == "doc_utile_saute"


def test_evaluate_returns_shares_that_sum_to_one():
    res = evaluate(_ctl(), list(iter_cases(11, 6)))
    assert abs(sum(res["classes"].values()) - 1.0) < 1e-6
    assert 0.0 <= res["exactitude"] <= 1.0
    assert res["n"] == 6 and res["octets_lus_moyen"] >= 0
    assert "exact" in format_report(res)


def _doc_case(needed_doc):
    """Cas minimal « document nécessaire » construit sans modèle : `classify` ne lit
    que `case.passes_needed[0][1]` (et, hors branche documents, `case.stop_index`)."""
    return Case(kind="doc_lookup", query=b"q", history=[], docs=[], doc_relevant=[],
                passes_needed=[(set(), needed_doc)], answer_parts=[b"reponse"], notes=[],
                answer_value="x")


def test_classify_doc_never_reached():
    # Le balayage s'est arrêté DANS les documents (stop_at == -1) mais avant
    # d'atteindre l'index utile : `doc_actions` est trop court pour l'index 2.
    # Arrêté trop tôt (pas encore lu ce qu'il fallait), pas « sauté » : le canal
    # documents n'a même pas été ouvert jusqu'au bon document (spec §9, F3).
    c = _doc_case(2)
    assert classify(c, {"stop_at": -1, "doc_actions": ["read"]}) == "trop_tot"


def test_classify_doc_stopped_in_history():
    # Le balayage s'est arrêté dans l'HISTORIQUE (stop_at != -1) : les documents
    # n'ont jamais été ouverts, donc le document utile n'a jamais été lu — trop tôt,
    # pas « sauté » (le canal documents n'a même pas été ouvert, spec §9, F3).
    c = _doc_case(2)
    assert classify(c, {"stop_at": 1, "doc_actions": []}) == "trop_tot"


def test_classify_doc_exact_and_too_late():
    # L'indice d'arrivée (len(doc_actions) - 1) distingue « arrêté juste après avoir
    # lu le document utile » de « a continué à lire au-delà » : `stop_at` seul (-1
    # dans les deux cas) ne permet pas cette distinction.
    c = _doc_case(1)
    assert classify(c, {"stop_at": -1, "doc_actions": ["skip", "read"]}) == "exact"
    assert classify(c, {"stop_at": -1, "doc_actions": ["skip", "read", "read"]}) == "trop_tard"


def test_classify_doc_needed_but_skipped():
    c = _doc_case(1)
    assert classify(c, {"stop_at": -1, "doc_actions": ["skip", "skip", "read"]}) == "doc_utile_saute"


def _doc_seg(name="d.txt"):
    return Segment(header=format_header({"type": "doc", "name": name}), content=b"x")


def test_classify_absent_with_docs():
    # Cas « absent » (F2) : aucun document nécessaire (`needed_doc is None`), mais
    # des documents existent quand même — l'oracle les visite TOUS avant de
    # s'arrêter au dernier (`stop_index == -1`). Un arrêt dans l'historique n'a
    # rien vérifié : trop tôt, pas « trop tard » (`stop_at >= 0 > -1` ne doit pas
    # suffire). Un arrêt au premier document sur trois n'a pas non plus tout
    # vérifié : trop tôt aussi, pas « exact » (`stop_at == -1 == stop_index` ne
    # doit pas suffire sans avoir visité les trois).
    c = Case(kind="absent", query=b"q", history=[], docs=[_doc_seg("a"), _doc_seg("b"), _doc_seg("c")],
             doc_relevant=[False, False, False], passes_needed=[(set(), None)],
             answer_parts=[b"absent"], notes=[], answer_value="")
    c.stop_index = -1
    assert classify(c, {"stop_at": 1, "doc_actions": []}) == "trop_tot"
    assert classify(c, {"stop_at": -1, "doc_actions": ["skip"]}) == "trop_tot"
    assert classify(c, {"stop_at": -1, "doc_actions": ["skip", "skip", "skip"]}) == "exact"


def test_needle_case_reaches_requested_size_and_keeps_the_fact():
    c = build_case(random.Random(1), 6000)
    total = sum(len(s.content) for s in c.history)
    assert 4000 <= total <= 12000
    joined = b" ".join(s.content for s in c.history).decode("utf-8", "replace")
    assert c.answer_value in joined
    assert 0 <= c.stop_index < len(c.history)


def test_needle_evaluate_returns_one_row_per_size():
    rows = needle_eval(_ctl(), sizes=(600, 1200), n_per_size=2, seed=5)
    assert [r["taille"] for r in rows] == [600, 1200]
    assert all(0.0 <= r["exactitude"] <= 1.0 for r in rows)
    assert "taille" in format_table(rows)


def test_needle_report_mentions_budget_stops():
    rows = needle_eval(_ctl(), sizes=(600,), n_per_size=2, seed=5)
    assert "budget" in format_table(rows)


def test_budget_for_covers_the_whole_case():
    # Un cas à 3 000 segments d'historique (bien au-delà du plafond d'entraînement
    # `Budget.max_segments == 2000`) doit recevoir un budget qui les couvre tous,
    # sinon le contrôleur est forcé SKIP puis STOP avant même d'avoir pu atteindre
    # le fait (spec §9, F1).
    hist = [Segment(header=format_header({"role": "user", "i": i}), content=b"x" * 20)
            for i in range(3_000)]
    case = Case(kind="needle", query=b"q", history=hist, docs=[], doc_relevant=[],
                passes_needed=[({1}, None)], answer_parts=[b"y"], notes=[], answer_value="y")
    b = budget_for(case, max_gen=64)
    assert b.max_segments > 3_000
    assert b.max_read_bytes >= sum(len(s.content) for s in hist)
    assert b.max_gen_bytes == 64


def test_summarise_reports_budget_stop():
    budget_stop = [Event("turn_start", text="q"), Event("scan_end", text="budget")]
    normal_stop = [Event("turn_start", text="q"), Event("scan_end", text="stop")]
    assert summarise(budget_stop)["stopped_by_budget"] is True
    assert summarise(normal_stop)["stopped_by_budget"] is False


class _FakeCtl:
    """Contrôleur factice pour tester `run_case` sans modèle : seul `.budget`
    (lu/écrit par `run_case`) et `.turn(query, history, docs)` sont utilisés."""

    def __init__(self, events):
        self.budget = Budget()
        self._events = events

    def turn(self, query, history, docs):
        return iter(self._events)


def test_saved_uses_the_first_pass_only():
    # Première passe : 100 octets lus sur 400 disponibles. REFRESH. Seconde passe :
    # relit les mêmes 100 octets (coût total cumulé = 200). `saved` doit rester basé
    # sur la première passe seule (100/400 -> 0.75), jamais sur le total, qui donnerait
    # une économie négative pour un modèle qui n'a pourtant rien lu de plus au fond
    # (spec §9, F4).
    events = [
        Event("turn_start", text="q"),
        Event("channel", channel="history", size=1),
        Event("segment", channel="history", header="h0", size=100, action="read"),
        Event("decision", channel="history", text="STOP"),
        Event("scan_end", text="stop"),
        Event("gen_byte", byte=65),
        Event("note", text=""),
        Event("refresh", index=1),
        Event("channel", channel="history", size=1),
        Event("segment", channel="history", header="h0b", size=100, action="read"),
        Event("decision", channel="history", text="STOP"),
        Event("scan_end", text="stop"),
        Event("end"),
        Event("turn_end", text="A", stats={"read": 200, "written": 1, "refresh": 1,
                                           "saved": -1.0, "seconds": 0.01, "available": 400}),
    ]
    case = Case(kind="fact_recall", query=b"q", history=[], docs=[], doc_relevant=[],
                passes_needed=[(set(), None)], answer_parts=[b"a"], notes=[], answer_value="a")
    r = run_case(_FakeCtl(events), case)
    assert r["read"] == 200                 # coût total, cumulé sur les deux passes
    assert r["saved"] == 0.75                # mais l'économie ne regarde que la 1re passe


def test_adaptive_returns_two_populations_and_a_ratio():
    res = adaptive_eval(_ctl(), n=3, seed=5)
    assert res["faciles"] >= 0 and res["difficiles"] >= 0
    assert res["rapport"] > 0
