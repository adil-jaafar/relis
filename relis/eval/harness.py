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


def budget_for(case, max_gen: int) -> Budget:
    """Budget dimensionné pour CE cas (spec §9, F1).

    `Budget.max_segments` par défaut (2 000) vaut pour des tours de remplissage
    d'entraînement d'une vingtaine d'octets ; un cas d'évaluation à 64 ko ou 200 ko
    en compte des milliers, et se ferait forcer SKIP puis STOP bien avant d'avoir
    pu atteindre le fait. On dimensionne donc le budget sur la taille réelle du
    cas plutôt que sur la constante d'entraînement : un segment par tour d'historique
    et par document, plus deux (marge pour CHAN/DEC), et au moins la totalité des
    octets de contenu (plus une marge) pour la lecture. `max_gen_bytes` reste celui
    passé en ligne de commande : lui seul borne la génération, pas la lecture.
    """
    content_bytes = (sum(len(s.content) for s in case.history) +
                      sum(len(s.content) for s in case.docs))
    return Budget(max_segments=len(case.history) + len(case.docs) + 2,
                  max_read_bytes=content_bytes + 4_096,
                  max_gen_bytes=max_gen)


def summarise(events) -> dict:
    """Agrège les compteurs dérivés du balayage sur la PREMIÈRE passe seulement.

    Un REFRESH relance `scan()` depuis le début sur le même historique/documents
    (spec §7.1) : sans arrêt à la fin de la première passe, `hist_segments` et
    `doc_actions` mélangeraient plusieurs passes — `stop_at` serait calculé sur un
    compte de segments gonflé par les passes suivantes, et `doc_actions` perdrait
    sa correspondance 1:1 avec les indices de `case.docs`. On s'arrête donc au
    premier `scan_end`, qui clôt exactement la passe que décrit `case.stop_index`
    (que cette passe se termine par STOP ou par épuisement des canaux).

    En plus des compteurs existants, deux champs :
    - `stopped_by_budget` : vrai si CETTE passe s'est arrêtée parce qu'un plafond de
      `Budget` a été atteint (`scan_end.text == "budget"`), pas parce que le modèle
      a lui-même décidé STOP — un signal que le chiffre principal est suspect (F1).
    - `read_first_pass` : somme des tailles des segments LUS pendant cette seule
      première passe, gelée avant toute relecture — contrairement à `read` (le
      coût total, cumulé sur toutes les passes), c'est la bonne base pour calculer
      une économie par rapport à `available` qui, lui, ne compte qu'une fois (F4).
    """
    hist_segments = []
    doc_actions = []
    stop_at = None
    stopped_by_budget = False
    read_first_pass = 0
    for e in events:
        if e.kind == "segment":
            if e.channel == "history":
                hist_segments.append(e)
            else:
                doc_actions.append(e.action)
            if e.action == "read":
                read_first_pass += e.size
        elif e.kind == "decision" and e.text == "STOP":
            stop_at = (len(hist_segments) - 1) if e.channel == "history" else -1
        elif e.kind == "scan_end":
            stopped_by_budget = e.text == "budget"
            break
    return {"stop_at": stop_at,
            "read_segments": sum(1 for e in hist_segments if e.action == "read"),
            "skipped_docs": sum(1 for a in doc_actions if a == "skip"),
            "doc_actions": doc_actions,
            "stopped_by_budget": stopped_by_budget,
            "read_first_pass": read_first_pass}


def run_case(ctl: Controller, case, budget: Budget | None = None) -> dict:
    """Le contrôleur voit l'historique et les documents **complets** et décide seul.

    `budget`, si fourni (typiquement `budget_for(case, max_gen)`), remplace
    temporairement `ctl.budget` pour ce seul tour, puis le budget d'origine est
    restauré — même si `ctl.turn` lève, pour ne jamais laisser le contrôleur dans
    un état inattendu pour l'appelant suivant.
    """
    prev = ctl.budget
    if budget is not None:
        ctl.budget = budget
    try:
        events = list(ctl.turn(case.query, case.history, case.docs))
    finally:
        ctl.budget = prev
    stats = events[-1].stats
    s = summarise(events)
    available = stats["available"]
    # `saved` se calcule sur `read_first_pass`, pas sur `stats["read"]` (le coût total,
    # cumulé sur toutes les passes) : sinon une relecture après REFRESH fait paraître
    # négative une économie qui n'a de sens que rapportée à la première passe (F4).
    saved = round(1.0 - s["read_first_pass"] / available, 3) if available else 0.0
    return {"answer": answer_of(events), "read": stats["read"], "written": stats["written"],
            "refresh": stats["refresh"], "saved": saved, "seconds": stats["seconds"],
            "events": events, **s}
