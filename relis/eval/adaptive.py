"""Calcul adaptatif : le coût suit-il la difficulté ? (spec §9)

Facile = le fait est dans les trois segments les plus récents, le modèle devrait
s'arrêter presque tout de suite. Difficile = le fait est dans le tiers le plus ancien,
ou la réponse est dans un document parmi plusieurs. On compare les octets lus.
"""
import argparse
import random

from relis.data.episodes import FACTS, Case, build_history, case_to_spec
from .harness import judge, load_controller, run_case


def _placed_case(rng: random.Random, recent: bool) -> Case:
    f = rng.choice(FACTS); v = f[4](rng)
    n = rng.randint(8, 14)
    at = rng.randint(n - 3, n - 1) if recent else rng.randint(0, max(0, n // 3))
    hist = build_history(rng, n, {at: f[0].format(v=v)})
    idx = 2 * (n - 1 - at) + 1
    case = Case(kind="adaptive", query=f[1].encode("utf-8"), history=hist, docs=[],
                doc_relevant=[], passes_needed=[({idx}, None)],
                answer_parts=[f[2].format(v=v).encode("utf-8")], notes=[], answer_value=str(v))
    case_to_spec(case)
    return case


def evaluate(ctl, n: int = 40, seed: int = 777) -> dict:
    rng = random.Random(seed)
    out = {}
    for nom, recent in (("faciles", True), ("difficiles", False)):
        lus, bons = 0, 0
        for _ in range(n):
            case = _placed_case(rng, recent)
            r = run_case(ctl, case)
            lus += r["read"]; bons += int(judge(case, r["answer"]))
        out[nom] = lus // n
        out[f"exactitude_{nom}"] = bons / n
    out["rapport"] = round(out["difficiles"] / max(1, out["faciles"]), 2)
    return out


def format_report(res: dict) -> str:
    return (f"faciles {res['faciles']} octets lus (justes {res['exactitude_faciles']:.3f}) · "
            f"difficiles {res['difficiles']} octets lus (justes {res['exactitude_difficiles']:.3f}) · "
            f"rapport ×{res['rapport']}")


def main(argv=None):
    import sys
    if hasattr(sys.stdout, "reconfigure"):          # consoles Windows en cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo"); ap.add_argument("--ckpt")
    ap.add_argument("--n", type=int, default=40)
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
    from relis.infer.controller import Budget
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ctl, step = load_controller(args.ckpt or args.repo, device, Budget(max_gen_bytes=args.max_gen))
    print(f"[adaptatif] checkpoint au pas {step}")
    print(format_report(evaluate(ctl, args.n, args.seed)))


if __name__ == "__main__":
    main()
