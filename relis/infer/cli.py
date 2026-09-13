"""Conversation en ligne de commande (spec §8).

    python -m relis.infer.cli --repo jaafar2022/relis-v1-tape --ask "Quelle est l'adresse ?"
    python -m relis.infer.cli --ckpt runs/tape1/last.pt --doc notes.md --doc api.py \\
        --conversation conv.json --ask "Où est le serveur ?"

Attention (F6) : les épisodes d'entraînement comportent toujours au moins un tour
d'historique ; un premier tour de conversation (sans `--conversation` existant, ou
avec un fichier vide), avec ou sans `--doc` joint, est donc hors distribution — y
compris la commande d'exemple ci-dessus. Pour une démonstration représentative,
poser d'abord une question anodine, laisser `--conversation` l'enregistrer, puis
poser la vraie question dans un second appel.
"""
import argparse
import datetime as _dt
import json
import mimetypes
import os
import sys

import torch

from relis.tape.headers import format_header
from relis.tape.tape import Segment
from relis.infer.controller import Budget, Controller
from relis.infer.render import TerminalRenderer
from relis.infer.sample import load_model


def _stamp() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%dT%H:%M")


def load_docs(paths) -> list:
    """Documents joints, du plus récemment joint au plus ancien (spec §4.3)."""
    segs = []
    for p in paths:
        with open(p, "rb") as f:
            content = f.read()
        mime = mimetypes.guess_type(p)[0] or "text/plain"
        segs.append(Segment(header=format_header({"type": "doc", "name": os.path.basename(p),
                                                  "mime": mime, "bytes": len(content), "t": _stamp()}),
                            content=content))
    return segs[::-1]


def history_segments(turns) -> list:
    """Tours passés, du plus récent au plus ancien."""
    segs = []
    for t in reversed(turns):
        fields = {"role": t["role"]}
        if t.get("t"):
            # Clé `t` absente plutôt que vide (F7) : `t=` sans valeur est une forme
            # jamais vue à l'entraînement, où l'horodatage est soit renseigné, soit
            # tout simplement omis de l'en-tête.
            fields["t"] = t["t"]
        segs.append(Segment(header=format_header(fields), content=t["text"].encode("utf-8")))
    return segs


def _load_conversation(path):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("turns", [])
    if path:
        print(f"[relis] conversation introuvable : {path} — on part d'un historique vide", file=sys.stderr)
    return []


def _warn_if_empty(turns) -> None:
    """Un historique vide est hors distribution (spec §9, F6) : le dire à l'écran plutôt que
    laisser le lecteur prendre une récitation du gabarit d'entraînement pour une réponse."""
    if not turns:
        print("[relis] historique vide : premier tour, hors distribution de l'entraînement ; "
              "les décisions et la réponse ne sont pas représentatives (voir demo/README.md)",
              file=sys.stderr)


def _save_conversation(path, turns):
    if not path:
        return
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"turns": turns}, f, ensure_ascii=False, indent=1)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo", help="dépôt Hugging Face contenant last.pt")
    src.add_argument("--ckpt", help="chemin local vers last.pt")
    ap.add_argument("--ask", help="une seule question ; sinon boucle interactive")
    ap.add_argument("--doc", action="append", default=[], help="document joint (répétable)")
    ap.add_argument("--conversation", help="fichier JSON de conversation persistante")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_gen", type=int, default=1024)
    ap.add_argument("--max_read", type=int, default=2_000_000)
    ap.add_argument("--max_refresh", type=int, default=8)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-color", dest="color", action="store_false", default=None)
    args = ap.parse_args(argv)

    path = args.ckpt
    if path is None:
        from relis.train.loop import fetch_weights_path
        path = fetch_weights_path(args.repo)
    model, step = load_model(path, args.device)
    if not args.quiet:
        print(f"[relis] checkpoint au pas {step} · {args.device}")

    ctl = Controller(model, args.device,
                     Budget(max_gen_bytes=args.max_gen, max_read_bytes=args.max_read,
                            max_refresh=args.max_refresh))
    docs = load_docs(args.doc)
    turns = _load_conversation(args.conversation)

    def one(question: str) -> None:
        _warn_if_empty(turns)
        events = ctl.turn(question.encode("utf-8"), history_segments(turns), docs)
        answer = TerminalRenderer(sys.stdout, args.color, args.quiet).run(events)
        turns.append({"role": "user", "text": question, "t": _stamp()})
        turns.append({"role": "assistant", "text": answer, "t": _stamp()})
        _save_conversation(args.conversation, turns)

    if args.ask:
        one(args.ask)
        return
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if q in ("", "/quit", "/exit"):
            return
        one(q)


if __name__ == "__main__":
    main()
