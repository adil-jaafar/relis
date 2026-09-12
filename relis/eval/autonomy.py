"""Décisions en autonomie contre l'oracle (spec §9).

En forçage, chaque décision est jugée en supposant correctes celles qui la précèdent.
Ici le contrôleur décide seul : une erreur de STOP change tout ce qu'il voit ensuite.
C'est la seule mesure qui dit si le mécanisme survit à la composition de ses erreurs.
"""
import argparse
import sys
from collections import Counter

from relis.data.episodes import iter_cases
from .harness import judge, load_controller, run_case

CLASSES = ("exact", "trop_tot", "trop_tard", "doc_utile_saute", "jamais_stop")


def classify(case, result) -> str:
    """Classe une décision d'arrêt contre l'oracle du cas, à partir de
    `result["stop_at"]` et `result["doc_actions"]` (produits par `harness.run_case`).

    `doc_actions` — pas un simple compteur de documents sautés — est nécessaire car un
    document nécessaire lu *aux côtés* d'autres documents sautés serait sinon confondu
    avec un document nécessaire réellement sauté : seul l'index utile compte, pas le
    nombre de sauts dans l'ensemble du passage. Cet index (`case.passes_needed[0][1]`)
    est absent de `doc_actions` (jamais atteint car le balayage s'est arrêté avant, dans
    l'historique ou dans les documents) tout autant que présent-mais-sauté : les deux
    situations valent « document utile sauté ».
    """
    if result["stop_at"] is None:
        return "jamais_stop"
    needed_doc = case.passes_needed[0][1]
    if needed_doc is not None:
        doc_actions = result["doc_actions"]
        if needed_doc >= len(doc_actions) or doc_actions[needed_doc] == "skip":
            return "doc_utile_saute"
    if result["stop_at"] == case.stop_index:
        return "exact"
    return "trop_tot" if result["stop_at"] < case.stop_index else "trop_tard"


def evaluate(ctl, cases) -> dict:
    counts, bons, lus, saved = Counter(), 0, 0, 0.0
    for case in cases:
        r = run_case(ctl, case)
        counts[classify(case, r)] += 1
        bons += int(judge(case, r["answer"]))
        lus += r["read"]
        saved += r["saved"]
    n = max(1, sum(counts.values()))
    return {"n": sum(counts.values()),
            "classes": {k: counts.get(k, 0) / n for k in CLASSES},
            "effectifs": {k: counts.get(k, 0) for k in CLASSES},
            "exactitude": bons / n, "octets_lus_moyen": lus // n,
            "economie_moyenne": round(saved / n, 3)}


def format_report(res: dict) -> str:
    cl = " ".join(f"{k} {res['effectifs'][k]}" for k in CLASSES)
    return (f"autonomie n={res['n']} | réponses justes {res['exactitude']:.3f} | "
            f"{cl} | {res['octets_lus_moyen']} octets lus en moyenne | "
            f"{int(res['economie_moyenne'] * 100)} % économisés")


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):          # consoles Windows en cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo"); ap.add_argument("--ckpt")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ctl, step = load_controller(args.ckpt or args.repo, device)
    print(f"[autonomie] checkpoint au pas {step}")
    print(format_report(evaluate(ctl, list(iter_cases(args.seed, args.n)))))


if __name__ == "__main__":
    main()
