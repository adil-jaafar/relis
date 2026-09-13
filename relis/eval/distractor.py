"""Sonde « distracteur » : le STOP suit-il la question, ou la simple forme d'un fait ?

Aucune famille d'épisodes d'entraînement ne place dans l'historique un fait qui ne répond PAS à la
question. Un modèle qui aurait appris « s'arrêter sur la première phrase en forme de fait » aurait
donc exactement les mêmes scores qu'un modèle qui compare le fait à la question. Quatre cas construits
à la main séparent les deux hypothèses ; la réponse attendue est indiquée pour chaque cas.

    python -m relis.eval.distractor --repo jaafar2022/relis-v1-tape
"""
import argparse
import time

from relis.data.episodes import Case
from relis.eval.harness import load_controller, run_case, budget_for, judge
from relis.tape.headers import format_header
from relis.tape.tape import Segment

ROOM = "La réunion aura lieu en salle C7."
SERVER = "L'adresse du serveur est 10.42.7.15."
Q_ROOM = "Dans quelle salle a lieu la réunion ?"
Q_SERVER = "Quelle est l'adresse du serveur ?"
FILL = [("Parlons du budget.", "Bien sûr."), ("J'ai avancé sur le planning ce matin.", "Très bien, continuons."),
        ("Merci pour ton aide.", "Je reste disponible."), ("On en reparle plus tard.", "C'est noté.")]


def _turn(role, text, i):
    return Segment(header=format_header({"role": role, "t": f"2026-09-13T09:{i:02d}"}), content=text.encode())


def _hist(pairs):
    """pairs du plus ancien au plus récent ; renvoie du plus récent au plus ancien, comme le contrôleur."""
    segs = []
    for i, (u, a) in enumerate(pairs):
        segs.append(_turn("user", u, 2 * i))
        segs.append(_turn("assistant", a, 2 * i + 1))
    return segs[::-1]


def _doc(name, body, i):
    b = body.encode()
    return Segment(header=format_header({"type": "doc", "name": name, "mime": "text/plain",
                                         "bytes": len(b), "t": f"2026-09-13T10:{i:02d}"}), content=b)


def _case(kind, query, pairs, docs, value, expect):
    c = Case(kind=kind, query=query.encode(), history=_hist(pairs), docs=docs,
             doc_relevant=[True] * len(docs), passes_needed=[(set(), None)],
             answer_parts=[b""], notes=[], answer_value=value)
    c.expect = expect
    return c


def cases():
    # index de segment dans l'ordre du balayage (0 = tour le plus récent)
    return [
        _case("A distracteur récent, réponse ancienne", Q_ROOM,
              [(ROOM, "C'est noté."), FILL[0], (SERVER, "C'est noté."), FILL[1]], [], "C7",
              "STOP au segment 7 (la salle), pas au segment 3 (le serveur)"),
        _case("B symétrique", Q_SERVER,
              [(SERVER, "C'est noté."), FILL[0], (ROOM, "C'est noté."), FILL[1]], [], "10.42.7.15",
              "STOP au segment 7 (le serveur), pas au segment 3 (la salle)"),
        _case("C document, historique anodin", Q_ROOM, FILL,
              [_doc("notes_3.txt", "Merci pour ton aide. Parlons du planning. On en reparle plus tard.", 0),
               _doc("reunion_12.txt", "Entendu, je m'en occupe. " + ROOM + " Merci pour ton aide.", 1)][::-1], "C7",
              "aucun STOP en historique, notes_3 sauté, reunion_12 lu"),
        _case("D distracteur puis document", Q_ROOM,
              [FILL[0], (SERVER, "C'est noté."), FILL[1], FILL[2]],
              [_doc("notes_3.txt", "Merci pour ton aide. Parlons du planning.", 0),
               _doc("reunion_12.txt", "Entendu. " + ROOM + " Merci.", 1)][::-1], "C7",
              "aucun STOP en historique malgré le fait serveur ; reunion_12 lu"),
    ]


def evaluate(ctl, max_gen=128):
    out = []
    for c in cases():
        t = time.time()
        r = run_case(ctl, c, budget=budget_for(c, max_gen))
        out.append({"kind": c.kind, "expect": c.expect, "stop_at": r["stop_at"], "doc_actions": r["doc_actions"],
                    "answer": r["answer"], "correct": judge(c, r["answer"]), "seconds": time.time() - t})
    return out


def format_report(rows):
    lines = []
    for r in rows:
        lines.append(f"{r['kind']}\n    attendu : {r['expect']}\n    obtenu  : stop_at={r['stop_at']} "
                     f"docs={r['doc_actions']} juste={r['correct']} réponse={r['answer']!r} ({r['seconds']:.0f} s)")
    return "\n".join(lines)


def main(argv=None):
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo"); ap.add_argument("--ckpt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max_gen", type=int, default=128)
    args = ap.parse_args(argv)
    if not (args.repo or args.ckpt):
        ap.error("--repo ou --ckpt est requis")
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ctl, step = load_controller(args.ckpt or args.repo, device)
    print(f"[distracteur] checkpoint au pas {step}")
    print(format_report(evaluate(ctl, args.max_gen)))


if __name__ == "__main__":
    main()
