"""Décisions en autonomie contre l'oracle (spec §9).

En forçage, chaque décision est jugée en supposant correctes celles qui la précèdent.
Ici le contrôleur décide seul : une erreur de STOP change tout ce qu'il voit ensuite.
C'est la seule mesure qui dit si le mécanisme survit à la composition de ses erreurs.
"""
import argparse
import sys
from collections import Counter

from relis.data.episodes import iter_cases
from .harness import budget_for, judge, load_controller, run_case

CLASSES = ("exact", "trop_tot", "trop_tard", "doc_utile_saute", "jamais_stop")


def classify(case, result) -> str:
    """Classe une décision d'arrêt contre l'oracle du cas, à partir de
    `result["stop_at"]` et `result["doc_actions"]` (produits par `harness.run_case`).

    Pour un cas sans document nécessaire, `stop_at` (position dans l'historique) se
    compare directement à `case.stop_index` — SAUF si des documents existent quand
    même (cas « absent » : aucun document n'est requis, mais l'oracle les visite
    tous avant de s'arrêter au dernier, `stop_index == -1`, spec F2). Dans ce cas
    précis, un arrêt dans l'historique (`stop_at >= 0`) serait à tort classé
    « trop_tard » par la seule comparaison à `stop_index`, alors que le contrôleur a
    lu TROP PEU (jamais ouvert les documents) ; et un arrêt au premier document sur
    six donnerait `stop_at == -1 == case.stop_index`, donc « exact », alors qu'aucun
    document n'a été vérifié. Il faut donc vérifier explicitement que TOUS les
    documents ont été visités (`len(doc_actions) == len(case.docs)`).

    Pour un cas AVEC document nécessaire, `stop_at` ne dit que « le balayage s'est
    arrêté dans les documents » (toujours -1, quel que soit le document en cause —
    `harness.summarise` ne distingue pas les documents entre eux) : il faut donc
    l'indice d'arrivée réel, tiré de la longueur de `doc_actions`
    (`len(doc_actions) - 1`, le dernier document ouvert avant le STOP), pour savoir
    si le contrôleur s'est arrêté PILE sur le document utile ou s'il a continué à en
    lire au-delà. Un arrêt avant même d'atteindre ce document (dans l'historique, ou
    dans les documents mais avant l'index utile) est un arrêt TROP TÔT, pas un
    document sauté (spec §9, F3) : `doc_utile_saute` est réservé au cas où le canal
    documents a bien été ouvert jusqu'au bon document, et que celui-ci a été
    explicitement SKIPpé.
    """
    if result["stop_at"] is None:
        return "jamais_stop"
    needed_doc = case.passes_needed[0][1]
    actions = result["doc_actions"]
    if needed_doc is None:                                   # aucun document nécessaire
        if not case.docs:
            if result["stop_at"] == case.stop_index:
                return "exact"
            return "trop_tot" if result["stop_at"] < case.stop_index else "trop_tard"
        # des documents existent quand même (ex. cas « absent ») : l'oracle les
        # visite tous, `exact` exige donc de les avoir tous visités aussi.
        if result["stop_at"] == -1 and len(actions) == len(case.docs):
            return "exact"
        return "trop_tot"
    if result["stop_at"] != -1:                              # arrêté avant d'ouvrir les documents
        return "trop_tot"
    if needed_doc >= len(actions):                           # documents ouverts, mais pas jusqu'au bon
        return "trop_tot"
    if actions[needed_doc] == "skip":
        return "doc_utile_saute"                             # atteint, puis explicitement sauté
    arret = len(actions) - 1                                 # document sur lequel le STOP est tombé
    if arret == needed_doc:
        return "exact"
    return "trop_tard"                                       # a continué après avoir lu ce qu'il fallait


def evaluate(ctl, cases, max_gen: int | None = None) -> dict:
    # `max_gen=None` reprend `ctl.budget.max_gen_bytes` : `budget_for` redimensionne
    # `max_segments`/`max_read_bytes` par cas dans tous les cas (F1), mais un appelant
    # qui a déjà configuré son propre plafond de génération (tests, scripts) n'a pas
    # à le voir remplacé par une constante d'évaluation choisie ailleurs.
    max_gen = ctl.budget.max_gen_bytes if max_gen is None else max_gen
    counts, bons, lus, saved, arrets_budget = Counter(), 0, 0, 0.0, 0
    for case in cases:
        r = run_case(ctl, case, budget=budget_for(case, max_gen))
        counts[classify(case, r)] += 1
        bons += int(judge(case, r["answer"]))
        lus += r["read"]
        saved += r["saved"]
        arrets_budget += int(r["stopped_by_budget"])
    n = max(1, sum(counts.values()))
    return {"n": sum(counts.values()),
            "classes": {k: counts.get(k, 0) / n for k in CLASSES},
            "effectifs": {k: counts.get(k, 0) for k in CLASSES},
            "exactitude": bons / n, "octets_lus_moyen": lus // n,
            "economie_moyenne": round(saved / n, 3),
            "arrets_budget": arrets_budget}


def format_report(res: dict) -> str:
    cl = " ".join(f"{k} {res['effectifs'][k]}" for k in CLASSES)
    n_budget = res["arrets_budget"]
    budget_note = f"arrêts par budget : {n_budget}"
    if n_budget:                     # le budget, pas le modèle, a tranché : chiffre suspect
        budget_note += " (chiffre principal suspect)"
    return (f"autonomie n={res['n']} | réponses justes {res['exactitude']:.3f} | "
            f"{cl} | {res['octets_lus_moyen']} octets lus en moyenne | "
            f"{int(res['economie_moyenne'] * 100)} % économisés | {budget_note}")


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):          # consoles Windows en cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo"); ap.add_argument("--ckpt")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max_gen", type=int, default=256,
                    help="plafond d'octets générés (Budget.max_gen_bytes) : un vrai "
                         "checkpoint qui n'émet jamais END ne doit pas faire tourner "
                         "l'évaluation indéfiniment")
    args = ap.parse_args(argv)
    if not (args.repo or args.ckpt):
        ap.error("--repo ou --ckpt est requis")
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ctl, step = load_controller(args.ckpt or args.repo, device)
    print(f"[autonomie] checkpoint au pas {step}")
    print(format_report(evaluate(ctl, list(iter_cases(args.seed, args.n)), args.max_gen)))


if __name__ == "__main__":
    main()
