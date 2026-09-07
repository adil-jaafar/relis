"""Construction du ruban (spec §4.3, §6.3) : quatre tableaux parallèles par position."""
from dataclasses import dataclass, field

from . import codes as C

W_ZERO, W_LOW, W_ONE, W_HIGH = 0, 1, 2, 3
WEIGHTS = (0.0, 0.1, 1.0, 5.0)
ENCODE, SCAN, GENERATE = int(C.Mode.ENCODE), int(C.Mode.SCAN), int(C.Mode.GENERATE)


@dataclass
class Segment:
    header: str
    content: bytes
    read: bool = True
    after: int = C.CONT            # CONT, STOP ou NEXT


@dataclass
class Pass:
    history: list                  # Segment, du plus récent au plus ancien
    docs: list | None = None       # None : canal documents non ouvert


@dataclass
class TapeSpec:
    query: bytes
    passes: list                   # Pass ; len >= 1
    answer_parts: list             # bytes, une par passe
    notes: list                    # bytes, une par REFRESH (len(passes) - 1)


@dataclass
class Tape:
    """Quatre tableaux parallèles, stockés en `bytearray` (1 octet par position).

    L'API de lecture reste celle d'une séquence d'entiers : `tape.mode[i]`,
    `tape.wclass[i]` sont des `int`, `tape.reset[i]` vaut 0 ou 1 (et `sum(tape.reset)`
    compte les réinitialisations). Le stockage compact est ce qui permet
    d'empaqueter 100 000 épisodes sans saturer la RAM (des listes Python
    coûteraient ≈ 27 octets par position, contre 4 ici).
    """

    data: bytearray = field(default_factory=bytearray)
    mode: bytearray = field(default_factory=bytearray)
    wclass: bytearray = field(default_factory=bytearray)
    reset: bytearray = field(default_factory=bytearray)

    def put(self, b: int, mode: int, w: int, reset: bool = False) -> None:
        self.data.append(b); self.mode.append(mode); self.wclass.append(w)
        self.reset.append(1 if reset else 0)

    def put_bytes(self, bs: bytes, mode: int, w: int) -> None:
        for b in bs:
            self.put(b, mode, w)

    def __len__(self) -> int:
        return len(self.data)

    def weights(self) -> list:
        return [WEIGHTS[w] for w in self.wclass]

    def decisions(self) -> list:
        return [b for b, w in zip(self.data, self.wclass) if w == W_HIGH]


def validate_spec(spec: TapeSpec) -> None:
    n = len(spec.passes)
    if n < 1:
        raise ValueError("au moins une passe")
    if len(spec.answer_parts) != n:
        raise ValueError("une réponse par passe")
    if len(spec.notes) != n - 1:
        raise ValueError("une note par REFRESH")
    for p in spec.passes:
        segs = list(p.history) + (list(p.docs) if p.docs is not None else [])
        for s in p.history[:-1]:
            if s.after not in (C.CONT, C.STOP):
                raise ValueError("NEXT n'est admis qu'au dernier segment de l'historique")
        if p.history and p.history[-1].after == C.NEXT and p.docs is None:
            raise ValueError("NEXT sans canal documents")
        if p.docs is not None and p.history and p.history[-1].after != C.NEXT:
            raise ValueError("le canal documents exige NEXT au dernier segment de l'historique")
        if p.docs is not None:
            for s in p.docs:
                if s.after not in (C.CONT, C.STOP):
                    raise ValueError("dans le canal documents : CONT ou STOP")
        for s in segs:
            if s.after not in (C.CONT, C.STOP, C.NEXT):
                raise ValueError("décision inconnue")
        if segs and segs[-1].after != C.STOP:
            raise ValueError("la dernière décision d'une passe doit être STOP")
        stops = [s for s in segs if s.after == C.STOP]
        if len(stops) > 1:
            raise ValueError("un seul STOP par passe")


def _segment(t: Tape, s: Segment, mode: int) -> None:
    t.put(C.SEG, mode, W_ZERO)
    t.put_bytes(s.header.encode("utf-8"), mode, W_LOW)
    t.put(C.HDR, mode, W_ZERO)
    t.put(C.READ if s.read else C.SKIP, mode, W_HIGH)
    if s.read:
        t.put_bytes(C.sanitize(s.content), mode, W_LOW)
    t.put(C.DEC, mode, W_ZERO)
    t.put(s.after, mode, W_HIGH)


def _channel(t: Tape, name: str, segs, mode: int) -> None:
    t.put(C.CHAN, mode, W_ZERO)
    t.put_bytes(f"chan={name}".encode("utf-8"), mode, W_LOW)
    t.put(C.HDR, mode, W_ZERO)
    for s in segs:
        _segment(t, s, mode)


def build_tape(spec: TapeSpec) -> Tape:
    validate_spec(spec)
    t = Tape()
    last = len(spec.passes) - 1
    for i, p in enumerate(spec.passes):
        t.put(C.ENC, ENCODE, W_ZERO, reset=(i > 0))
        t.put_bytes(C.sanitize(spec.query), ENCODE, W_LOW)
        if i > 0:
            t.put(C.PART, ENCODE, W_ZERO)
            t.put_bytes(C.sanitize(b"".join(spec.answer_parts[:i])), ENCODE, W_LOW)
            t.put(C.NOTE, ENCODE, W_ZERO)
            t.put_bytes(C.sanitize(spec.notes[i - 1]), ENCODE, W_LOW)
        t.put(C.SCAN, SCAN, W_ZERO)
        _channel(t, "history", p.history, SCAN)
        if p.docs is not None:
            _channel(t, "docs", p.docs, SCAN)
        t.put(C.GEN, GENERATE, W_ZERO)
        t.put_bytes(C.sanitize(spec.answer_parts[i]), GENERATE, W_ONE)
        if i < last:
            t.put(C.NOTE, GENERATE, W_HIGH)
            t.put_bytes(C.sanitize(spec.notes[i]), GENERATE, W_ONE)
            t.put(C.REFRESH, GENERATE, W_HIGH)
        else:
            t.put(C.END, GENERATE, W_HIGH)
    return t
