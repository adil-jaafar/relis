"""Harnais commun aux trois évaluations en autonomie (spec §9)."""
import os
import re

import torch

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
    return _norm(case.answer_value) in a


def run_case(ctl: Controller, case) -> dict:
    """Le contrôleur voit l'historique et les documents **complets** et décide seul."""
    events = list(ctl.turn(case.query, case.history, case.docs))
    stats = events[-1].stats
    hist_segments = [e for e in events if e.kind == "segment" and e.channel == "history"]
    doc_segments = [e for e in events if e.kind == "segment" and e.channel == "docs"]
    stop_at = None
    for e in events:
        if e.kind == "decision" and e.text == "STOP":
            stop_at = (len(hist_segments) - 1) if e.channel == "history" else -1
            break
    return {"answer": answer_of(events), "read": stats["read"], "written": stats["written"],
            "refresh": stats["refresh"], "saved": stats["saved"], "seconds": stats["seconds"],
            "stop_at": stop_at, "read_segments": sum(1 for e in hist_segments if e.action == "read"),
            "skipped_docs": sum(1 for e in doc_segments if e.action == "skip"),
            "doc_actions": [e.action for e in doc_segments], "events": events}
