"""Contrôleur d'inférence (spec §7.1) : la boucle qui fait décider le modèle.

Primitive unique : avaler des octets dans un mode donné et récupérer les logits de la
dernière position, qui prédisent l'octet suivant. Décider, c'est masquer ces logits aux
octets légaux et prendre le maximum. Le balayage passe par le mode bloc, la génération
par le mode pas-à-pas ; leur équivalence numérique est prouvée au plan 1.

Les cas limites (§4.7) sont câblés ici par restriction du masque, jamais appris : au
dernier segment de l'historique CONT vaut NEXT s'il existe des documents et STOP sinon,
au dernier document tout vaut STOP.
"""
from dataclasses import dataclass, field
import time

import torch

from relis.tape import codes as C
from .constrain import (AFTER_DEC_FULL, AFTER_HDR, GEN_CODES, NOTE_CODES,
                        Utf8Guard, mask_for)

MODE_ENC = int(C.Mode.ENCODE)
MODE_SCAN = int(C.Mode.SCAN)
MODE_GEN = int(C.Mode.GENERATE)


@dataclass
class Budget:
    """Quatre plafonds pour qu'un tour rende la main même si le modèle n'arrête jamais."""
    max_read_bytes: int = 2_000_000
    max_segments: int = 2_000
    max_refresh: int = 8
    max_gen_bytes: int = 4_096
    chunk: int = 4_096            # morceaux avalés en mode bloc
    max_note_bytes: int = 200


@dataclass
class Event:
    kind: str
    channel: str = ""
    header: str = ""
    size: int = 0
    action: str = ""
    byte: int = -1
    text: str = ""
    index: int = 0
    stats: dict = field(default_factory=dict)


def answer_of(events) -> str:
    """Réponse reconstruite depuis le flux d'événements."""
    return bytes(e.byte for e in events if e.kind == "gen_byte").decode("utf-8", "replace")


class Controller:
    def __init__(self, model, device: str = "cpu", budget: Budget | None = None):
        self.model = model.eval()
        self.device = device
        self.budget = budget or Budget()
        self.state = None
        self._masks: dict = {}

    # ---- primitives ----
    def _mask(self, allowed):
        m = self._masks.get(allowed)
        if m is None:
            m = mask_for(allowed, self.device)
            self._masks[allowed] = m
        return m

    @torch.no_grad()
    def _feed(self, data: bytes, mode: int):
        """Avale des octets en mode bloc ; renvoie les logits de la dernière position."""
        last = None
        step = self.budget.chunk
        for i in range(0, len(data), step):
            part = data[i:i + step]
            x = torch.tensor([list(part)], dtype=torch.long, device=self.device)
            m = torch.full_like(x, mode)
            logits, self.state = self.model(x, m, self.state)
            last = logits[0, -1]
        return last

    @torch.no_grad()
    def _feed_one(self, b: int, mode: int):
        x = torch.tensor([b], dtype=torch.long, device=self.device)
        m = torch.tensor([mode], dtype=torch.long, device=self.device)
        logits, self.state = self.model.step(x, m, self.state)
        return logits[0]

    def _decide(self, logits, allowed) -> int:
        return int((logits.float() + self._mask(allowed)).argmax())

    # ---- le tour ----
    def turn(self, query: bytes, history=(), docs=()):
        bud = self.budget
        t0 = time.time()
        self.state = self.model.new_state(1, self.device)
        query = C.sanitize(query)
        history, docs = list(history), list(docs)
        available = sum(len(s.content) for s in history) + sum(len(s.content) for s in docs)
        cnt = {"read": 0, "written": 0, "refresh": 0, "segments": 0, "available": available}
        answer = bytearray()
        pending = {"refresh": False, "note": b""}
        channels = [(n, s) for n, s in (("history", history), ("docs", docs)) if s]
        yield Event("turn_start", text=query.decode("utf-8", "replace"))

        def encode(partial: bytes, note: bytes, first: bool):
            self._feed(bytes([C.ENC]) + query, MODE_ENC)
            if not first:
                # Passe au-delà de la première (après un REFRESH) : PART puis NOTE
                # sont TOUJOURS émis, même vides. `GEN_CODES` permet REFRESH sans
                # passer par NOTE (note vide) : sans ce `first`, un REFRESH nu
                # donnerait `ENC query PART partiel SCAN` (NOTE et son marqueur
                # purement absents) au lieu de `ENC query PART partiel NOTE SCAN` —
                # une séquence jamais vue à l'entraînement (spec §9, F5).
                self._feed(bytes([C.PART]) + partial, MODE_ENC)
                self._feed(bytes([C.NOTE]) + note, MODE_ENC)

        def scan():
            self._feed(bytes([C.SCAN]), MODE_SCAN)
            for ci, (cname, segs) in enumerate(channels):
                more = ci + 1 < len(channels)
                yield Event("channel", channel=cname, size=len(segs))
                self._feed(bytes([C.CHAN]) + f"chan={cname}".encode("utf-8") + bytes([C.HDR]), MODE_SCAN)
                for si, seg in enumerate(segs):
                    cnt["segments"] += 1
                    over = cnt["read"] >= bud.max_read_bytes or cnt["segments"] > bud.max_segments
                    # SEG + en-tête et HDR fusionnés en un seul `_feed` (F7) : même
                    # mode, mêmes octets dans le même ordre qu'avant, juste sans la
                    # frontière d'appel artificielle entre les deux — un balayage de
                    # 64 ko économise ainsi des milliers de passes modèle inutiles
                    # (un en-tête tient presque toujours dans un seul chunk `_feed`).
                    lg = self._feed(bytes([C.SEG]) + seg.header.encode("utf-8") + bytes([C.HDR]), MODE_SCAN)
                    a = self._decide(lg, frozenset({C.SKIP}) if over else AFTER_HDR)
                    self._feed_one(a, MODE_SCAN)
                    if a == C.READ:
                        content = C.sanitize(seg.content)
                        if content:
                            self._feed(content, MODE_SCAN)
                        cnt["read"] += len(content)
                    yield Event("segment", channel=cname, header=seg.header,
                                size=len(seg.content), action="read" if a == C.READ else "skip")
                    lg = self._feed(bytes([C.DEC]), MODE_SCAN)
                    last = si + 1 == len(segs)
                    if over:
                        allowed = frozenset({C.STOP})
                    elif last and more:
                        allowed = frozenset({C.STOP, C.NEXT})
                    elif last:
                        allowed = frozenset({C.STOP})
                    elif more:
                        allowed = AFTER_DEC_FULL
                    else:
                        allowed = frozenset({C.CONT, C.STOP})
                    a = self._decide(lg, allowed)
                    self._feed_one(a, MODE_SCAN)
                    yield Event("decision", channel=cname, text=C.name(a))
                    if a == C.STOP:
                        yield Event("scan_end", text="budget" if over else "stop")
                        return
                    if a == C.NEXT:
                        break
            yield Event("scan_end", text="exhausted")

        def generate():
            pending["refresh"] = False
            pending["note"] = b""
            guard = Utf8Guard()
            lg = self._feed(bytes([C.GEN]), MODE_GEN)
            while True:
                # Le budget ne force l'arrêt qu'à une frontière de caractère : hors
                # frontière, le masque UTF-8 reste seul maître, quitte à dépasser le
                # budget d'au plus trois octets, le temps de terminer le caractère en cours.
                over_gen = cnt["written"] >= bud.max_gen_bytes
                if over_gen and guard.at_boundary:
                    allowed = frozenset({C.END})          # arrêt propre, à une frontière
                else:
                    allowed = guard.allowed()             # hors frontière : finir le caractère
                    if guard.at_boundary and not over_gen:
                        extra = {C.END}
                        if cnt["refresh"] < bud.max_refresh:
                            extra |= set(GEN_CODES) - {C.END}
                        allowed = allowed | frozenset(extra)
                b = self._decide(lg, allowed)
                if b == C.END:
                    self._feed_one(b, MODE_GEN)
                    yield Event("end")
                    return
                if b in (C.NOTE, C.REFRESH):
                    note = bytearray()
                    if b == C.NOTE:
                        lg = self._feed_one(b, MODE_GEN)
                        ng = Utf8Guard()
                        while True:
                            over_note = len(note) >= bud.max_note_bytes
                            if over_note and ng.at_boundary:
                                break                      # note complète, on force REFRESH
                            na = ng.allowed()
                            if ng.at_boundary and not over_note:
                                na = na | NOTE_CODES
                            nb = self._decide(lg, na)
                            if nb == C.REFRESH:
                                break
                            note.append(nb)
                            ng.feed(nb)
                            lg = self._feed_one(nb, MODE_GEN)
                        yield Event("note", text=bytes(note).decode("utf-8", "replace"))
                    self._feed_one(C.REFRESH, MODE_GEN)
                    self.state.reset_memory()
                    cnt["refresh"] += 1
                    pending["refresh"] = True
                    pending["note"] = bytes(note)
                    yield Event("refresh", index=cnt["refresh"])
                    return
                answer.append(b)
                cnt["written"] += 1
                guard.feed(b)
                yield Event("gen_byte", byte=b)
                lg = self._feed_one(b, MODE_GEN)

        encode(b"", b"", first=True)
        yield from scan()
        while True:
            yield from generate()
            if not pending["refresh"]:
                break
            encode(bytes(answer), pending["note"], first=False)
            yield from scan()

        cnt["seconds"] = round(time.time() - t0, 2)
        # Rien de disponible à lire : rien à économiser non plus, donc 0.0 (et pas 1.0).
        cnt["saved"] = round(1.0 - cnt["read"] / max(1, available), 3) if available else 0.0
        yield Event("turn_end", text=bytes(answer).decode("utf-8", "replace"), stats=dict(cnt))
