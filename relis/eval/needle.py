"""Aiguille : exactitude et coût selon la longueur de l'historique (spec §9).

Le fait est placé dans un historique qu'on rallonge jusqu'à la taille visée, bien
au-delà des longueurs vues à l'entraînement. Un Transformer de même taille s'effondre
dès que l'historique dépasse sa fenêtre ; RELIS n'a pas de fenêtre, la question est
donc de savoir si sa Mémoire tient.
"""
import argparse
import random

from relis.data.episodes import FACTS, Case, build_history, case_to_spec
from .harness import budget_for, judge, load_controller, run_case

# Garde-fou de la boucle de croissance : même pour la plus grande taille visée par les
# évaluations (200 000 octets), on ne construit jamais plus de NEEDLE_MAX_TOURS paires
# (soit 2 x NEEDLE_MAX_TOURS segments). Les tours de remplissage font une soixantaine
# d'octets en moyenne, donc quelques milliers de paires suffisent largement à dépasser
# 200 000 octets ; la borne est une ceinture de sécurité contre un tirage anormalement
# défavorable, pas le mécanisme qui atteint la taille visée en pratique.
NEEDLE_MAX_TOURS = 4_000


def build_case(rng: random.Random, target_bytes: int) -> Case:
    f = rng.choice(FACTS)
    v = f[4](rng)
    n = 4
    while True:
        at = rng.randint(0, n - 1)
        hist = build_history(rng, n, {at: f[0].format(v=v)})
        if sum(len(s.content) for s in hist) >= target_bytes or n >= NEEDLE_MAX_TOURS:
            break
        n = min(NEEDLE_MAX_TOURS, max(n + 2, int(n * 1.6)))
    idx = 2 * (n - 1 - at) + 1
    case = Case(kind="needle", query=f[1].encode("utf-8"), history=hist, docs=[],
                doc_relevant=[], passes_needed=[({idx}, None)],
                answer_parts=[f[2].format(v=v).encode("utf-8")], notes=[], answer_value=str(v))
    case_to_spec(case)                     # renseigne stop_index
    return case


def evaluate(ctl, sizes=(1_000, 4_000, 16_000, 64_000, 200_000), n_per_size: int = 20,
             seed: int = 777, max_gen: int | None = None):
    # `max_gen=None` reprend `ctl.budget.max_gen_bytes` (voir `autonomy.evaluate`).
    max_gen = ctl.budget.max_gen_bytes if max_gen is None else max_gen
    rows = []
    for size in sizes:
        rng = random.Random(seed + size)
        bons, lus, secs, arrets_budget = 0, 0, 0.0, 0
        for _ in range(n_per_size):
            case = build_case(rng, size)
            r = run_case(ctl, case, budget=budget_for(case, max_gen))
            bons += int(judge(case, r["answer"]))
            lus += r["read"]; secs += r["seconds"]
            arrets_budget += int(r["stopped_by_budget"])
        rows.append({"taille": size, "exactitude": bons / n_per_size,
                     "octets_lus_moyen": lus // n_per_size,
                     "secondes_moyennes": round(secs / n_per_size, 2),
                     "arrets_budget": arrets_budget})
    return rows


def format_table(rows) -> str:
    out = [f"{'taille':>9} {'exactitude':>11} {'octets lus':>11} {'secondes':>9}  arrêts par budget"]
    for r in rows:
        nb = r["arrets_budget"]
        note = f"arrêts par budget : {nb}"
        if nb:                      # le budget, pas le modèle, a tranché : ligne suspecte
            note += " (chiffre suspect)"
        out.append(f"{r['taille']:>9} {r['exactitude']:>11.3f} "
                   f"{r['octets_lus_moyen']:>11} {r['secondes_moyennes']:>9}  {note}")
    return "\n".join(out)


def main(argv=None):
    import sys
    if hasattr(sys.stdout, "reconfigure"):          # consoles Windows en cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo"); ap.add_argument("--ckpt")
    ap.add_argument("--sizes", default="1000,4000,16000,64000")
    ap.add_argument("--n", type=int, default=20)
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
    print(f"[aiguille] checkpoint au pas {step}")
    print(format_table(evaluate(ctl, tuple(int(s) for s in args.sizes.split(",")),
                                args.n, args.seed, args.max_gen)))


if __name__ == "__main__":
    main()
