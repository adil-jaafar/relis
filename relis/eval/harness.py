"""Harnais commun aux trois évaluations en autonomie (spec §9)."""
import os
import re

from relis.infer.controller import Budget, Controller, answer_of
from relis.infer.sample import load_model

_ABSENT = ("ne trouve pas", "ne sais pas", "aucune information", "pas cette information")


def load_controller(source: str, device: str = "cpu", budget: Budget | None = None):
    path = source
    if not os.path.isfile(path):
        from relis.train.loop import fetch_weights_path
        path = fetch_weights_path(source)
    model, step = load_model(path, device)
    return Controller(model, device, budget or Budget()), step


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def judge(case, answer: str) -> bool:
    a = _norm(answer)
    if not case.answer_value:
        return any(k in a for k in _ABSENT)
    pattern = r"(?<!\w)" + re.escape(_norm(case.answer_value)) + r"(?!\w)"
    return re.search(pattern, a) is not None


def summarise(events) -> dict:
    """Agrège les compteurs dérivés du balayage sur la PREMIÈRE passe seulement.

    Un REFRESH relance `scan()` depuis le début sur le même historique/documents
    (spec §7.1) : sans arrêt à la première décision STOP, `hist_segments` et
    `doc_actions` mélangeraient plusieurs passes — `stop_at` serait calculé sur un
    compte de segments gonflé par les passes suivantes, et `doc_actions` perdrait
    sa correspondance 1:1 avec les indices de `case.docs`. On s'arrête donc dès le
    premier STOP, qui correspond exactement à ce que décrit `case.stop_index`.
    """
    hist_segments = []
    doc_actions = []
    stop_at = None
    for e in events:
        if e.kind == "segment":
            if e.channel == "history":
                hist_segments.append(e)
            else:
                doc_actions.append(e.action)
        elif e.kind == "decision" and e.text == "STOP":
            stop_at = (len(hist_segments) - 1) if e.channel == "history" else -1
            break
    return {"stop_at": stop_at,
            "read_segments": sum(1 for e in hist_segments if e.action == "read"),
            "skipped_docs": sum(1 for a in doc_actions if a == "skip"),
            "doc_actions": doc_actions}


def run_case(ctl: Controller, case) -> dict:
    """Le contrôleur voit l'historique et les documents **complets** et décide seul."""
    events = list(ctl.turn(case.query, case.history, case.docs))
    stats = events[-1].stats
    return {"answer": answer_of(events), "read": stats["read"], "written": stats["written"],
            "refresh": stats["refresh"], "saved": stats["saved"], "seconds": stats["seconds"],
            "events": events, **summarise(events)}
