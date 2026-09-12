# RELIS — Plan 3 : le contrôleur, la CLI riche et les évaluations en autonomie

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Faire décider RELIS pour de vrai. Jusqu'ici ses décisions ne sont mesurées qu'en forçage, où chaque décision est jugée en supposant correctes toutes les précédentes. Ce plan livre la boucle d'inférence qui laisse le modèle piloter sa propre lecture, un terminal qui montre cette lecture en direct, et trois évaluations qui disent si le mécanisme survit à la composition de ses erreurs.

**Architecture:** Une primitive unique, « avaler des octets dans un mode et récupérer les logits de la dernière position ». Décider, c'est masquer ces logits aux octets légaux et prendre le maximum. Deux masques se combinent : la phase (§4.7) et un automate UTF-8 complet qui rend tout caractère invalide impossible. Le contrôleur émet un flux d'événements ; le rendu terminal et les évaluations ne sont que deux consommateurs de ce flux.

**Tech Stack:** Python ≥ 3.10, PyTorch ≥ 2.2, aucune dépendance nouvelle (couleurs en ANSI direct). Code existant : `relis/model`, `relis/tape`, `relis/data/episodes.py`, `relis/infer/sample.py`, `relis/train/loop.py`.

**Spec:** `docs/superpowers/specs/2026-09-06-relis-design.md` — §4.2, §4.3, §4.7, §4.8, §7.1, §7.2, §7.3, §8, §9.

## Global Constraints

- **RECALL est hors périmètre.** Les rubans d'entraînement n'émettent aucune cible RECALL, le modèle n'a jamais vu ce code : il ne doit jamais figurer dans un masque de décision.
- **Cas limites câblés, pas appris** (spec §7.1) : au dernier segment de l'historique, CONT vaut NEXT s'il existe des documents et STOP sinon ; au dernier document, CONT et NEXT valent STOP. Implémentés par restriction du masque, jamais par remappage après coup.
- **Budgets** : `max_read_bytes` 2 000 000, `max_segments` 2 000, `max_refresh` 8, `max_gen_bytes` 4 096. Un budget épuisé pendant le balayage force SKIP puis STOP ; pendant la génération, il force END.
- **REFRESH à l'inférence** = `state.reset_memory()` (état GDN à zéro, fenêtre locale vidée, `pending`, `seen` et slots conservés), puis ré-encodage `ENC query PART partiel NOTE note`. L'équivalence avec le masque d'entraînement est déjà prouvée par `tests/test_reset.py`.
- **Aucun caractère UTF-8 invalide ne peut être émis.** L'automate applique la table complète, plages restreintes comprises : `E0` → continuation `A0-BF`, `ED` → `80-9F`, `F0` → `90-BF`, `F4` → `80-8F`. Les codes de contrôle ne sont autorisés qu'à une frontière de caractère.
- Le balayage utilise `model.forward` (mode bloc), la génération `model.step` (mode pas-à-pas). Les deux sont numériquement équivalents, prouvé au plan 1 ; ne jamais générer octet par octet via `forward`, qui rembourre chaque octet à un chunk de 64.
- Tests sur processeur avec `RelisConfig.tiny()`, suite complète sous 60 secondes. Un test échoue avant le code qui le fait passer. Un commit par tâche au minimum.
- Branche `plan3-controleur`, créée depuis `main` à `a0ebbfc`. Ne pas toucher à `relis/train/`.

---

## Structure des fichiers

```
relis/infer/constrain.py     (créer) automate UTF-8 + ensembles de phase + masquage
relis/infer/controller.py    (créer) Budget, Event, Controller
relis/infer/render.py        (créer) rendu terminal ANSI du flux d'événements
relis/infer/cli.py           (créer) point d'entrée conversationnel
relis/data/episodes.py       (modifier) Case, generate_case, case_to_spec
relis/eval/__init__.py       (créer)
relis/eval/harness.py        (créer) charger un checkpoint, construire un contrôleur, juger une réponse
relis/eval/autonomy.py       (créer) décisions en autonomie contre l'oracle
relis/eval/needle.py         (créer) exactitude selon la longueur d'historique
relis/eval/adaptive.py       (créer) octets lus, facile contre difficile
notebooks/COLAB_EVAL.md      (créer) mode d'emploi des trois évaluations
tests/test_constrain.py, test_controller.py, test_render.py, test_cli.py,
tests/test_cases.py, test_eval.py   (créer)
```

---

### Task 1 : décodage contraint

**Files:** Create `relis/infer/constrain.py` ; Test `tests/test_constrain.py`

**Interfaces:**
- Produces : états `S0, S1, S2, S3, SA0, SED, SF0, SF4` ; `text_allowed(state) -> frozenset[int]` ; `advance(state, byte) -> int` (lève `ValueError` sur un octet illégal) ; `class Utf8Guard` avec `state`, `at_boundary`, `allowed()`, `feed(b)`, `reset()` ; ensembles de phase `AFTER_HDR`, `AFTER_DEC_FULL`, `GEN_CODES`, `NOTE_CODES` ; `mask_for(allowed, device) -> Tensor(256,)` (additif, 0 ou −inf) ; `masked_argmax(logits, allowed) -> int`.

- [ ] **Step 1 : tests**

`tests/test_constrain.py` :
```python
import random
import pytest
import torch
from relis.tape import codes as C
from relis.infer import constrain as K


ALL_STATES = (K.S0, K.S1, K.S2, K.S3, K.SA0, K.SED, K.SF0, K.SF4)


def test_allowed_matches_advance_on_every_byte():
    for st in ALL_STATES:
        allowed = K.text_allowed(st)
        for b in range(256):
            ok = True
            try:
                K.advance(st, b)
            except ValueError:
                ok = False
            assert ok == (b in allowed), (st, hex(b))


def test_random_walks_always_decode():
    rng = random.Random(0)
    for _ in range(40):
        g = K.Utf8Guard()
        out = bytearray()
        for _ in range(200):
            out.append(rng.choice(sorted(g.allowed())))
            g.feed(out[-1])
        while not g.at_boundary:                 # terminer le caractère en cours
            out.append(sorted(g.allowed())[0]); g.feed(out[-1])
        bytes(out).decode("utf-8")               # strict : lève si invalide


def test_restricted_ranges_reject_overlong_and_surrogates():
    assert 0x80 not in K.text_allowed(K.SA0) and 0xA0 in K.text_allowed(K.SA0)   # E0
    assert 0xA0 not in K.text_allowed(K.SED) and 0x9F in K.text_allowed(K.SED)   # ED, demi-codets
    assert 0x80 not in K.text_allowed(K.SF0) and 0x90 in K.text_allowed(K.SF0)   # F0
    assert 0x90 not in K.text_allowed(K.SF4) and 0x8F in K.text_allowed(K.SF4)   # F4


def test_illegal_lead_bytes_are_never_allowed():
    s0 = K.text_allowed(K.S0)
    for b in (0xC0, 0xC1, 0xF5, 0xFF, 0x80, 0xBF):
        assert b not in s0
    for b in C.CONTROL_CODES:
        assert b not in s0                       # les codes passent par le masque de phase
    for b in (0x09, 0x0A, 0x0D):
        assert b in s0


def test_guard_boundary_tracking():
    g = K.Utf8Guard()
    assert g.at_boundary
    g.feed(0xC3)
    assert not g.at_boundary
    g.feed(0xA9)                                 # « é »
    assert g.at_boundary


def test_phase_sets_exclude_recall():
    assert C.RECALL not in K.GEN_CODES and C.RECALL not in K.NOTE_CODES
    assert K.AFTER_HDR == frozenset({C.READ, C.SKIP})
    assert K.AFTER_DEC_FULL == frozenset({C.CONT, C.STOP, C.NEXT})
    assert K.GEN_CODES == frozenset({C.END, C.REFRESH, C.NOTE})
    assert K.NOTE_CODES == frozenset({C.REFRESH})


def test_masked_argmax_respects_allowed():
    logits = torch.zeros(256); logits[C.CONT] = 10.0; logits[C.STOP] = 5.0
    assert K.masked_argmax(logits, frozenset({C.STOP, C.NEXT})) == C.STOP
    assert K.masked_argmax(logits, K.AFTER_DEC_FULL) == C.CONT


def test_mask_is_additive_and_finite_where_allowed():
    m = K.mask_for(frozenset({C.END}), "cpu")
    assert m.shape == (256,) and m[C.END].item() == 0.0
    assert torch.isinf(m[C.CONT]) and m[C.CONT].item() < 0
```

- [ ] **Step 2 : vérifier l'échec** — `python -m pytest tests/test_constrain.py -q` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3 : implémenter**

`relis/infer/constrain.py` :
```python
"""Décodage contraint (spec §7.2) : masque de phase et automate UTF-8 complet.

Deux masques se combinent à chaque position. Le masque de phase dit quels codes de
contrôle sont légaux là où l'on est (§4.7). L'automate dit quels octets de texte
peuvent suivre, table UTF-8 complète comprise : les encodages trop longs et les
demi-codets sont impossibles, pas seulement rares.
"""
import torch

from relis.tape import codes as C

# États : S0 = frontière de caractère ; Sn = n continuations libres attendues ;
# SA0/SED/SF0/SF4 = première continuation à plage restreinte.
S0, S1, S2, S3, SA0, SED, SF0, SF4 = range(8)

TEXT_ASCII = frozenset({0x09, 0x0A, 0x0D}) | frozenset(range(0x20, 0x80))

_CONT_RANGE = {S1: (0x80, 0xBF), S2: (0x80, 0xBF), S3: (0x80, 0xBF),
               SA0: (0xA0, 0xBF), SED: (0x80, 0x9F), SF0: (0x90, 0xBF), SF4: (0x80, 0x8F)}
_CONT_NEXT = {S1: S0, S2: S1, S3: S2, SA0: S1, SED: S1, SF0: S2, SF4: S2}


def _lead_target(b: int):
    """État atteint après un octet de tête, ou None si ce n'est pas une tête valide."""
    if 0xC2 <= b <= 0xDF: return S1
    if b == 0xE0: return SA0
    if 0xE1 <= b <= 0xEC: return S2
    if b == 0xED: return SED
    if 0xEE <= b <= 0xEF: return S2
    if b == 0xF0: return SF0
    if 0xF1 <= b <= 0xF3: return S3
    if b == 0xF4: return SF4
    return None


def text_allowed(state: int) -> frozenset:
    """Octets de texte légaux dans cet état. Les codes de contrôle en sont exclus :
    ils relèvent du masque de phase, jamais de l'automate."""
    if state == S0:
        return frozenset(TEXT_ASCII | {b for b in range(0xC2, 0xF5) if _lead_target(b) is not None})
    lo, hi = _CONT_RANGE[state]
    return frozenset(range(lo, hi + 1))


def advance(state: int, b: int) -> int:
    if state == S0:
        if b in TEXT_ASCII:
            return S0
        t = _lead_target(b)
        if t is None:
            raise ValueError(f"octet illégal en frontière de caractère : {b:#04x}")
        return t
    lo, hi = _CONT_RANGE[state]
    if not lo <= b <= hi:
        raise ValueError(f"continuation illégale {b:#04x} dans l'état {state}")
    return _CONT_NEXT[state]


class Utf8Guard:
    """Suit l'état UTF-8 de la sortie en cours de génération."""

    def __init__(self):
        self.state = S0

    @property
    def at_boundary(self) -> bool:
        return self.state == S0

    def allowed(self) -> frozenset:
        return text_allowed(self.state)

    def feed(self, b: int) -> "Utf8Guard":
        self.state = advance(self.state, b)
        return self

    def reset(self) -> "Utf8Guard":
        self.state = S0
        return self


# Ensembles de phase (spec §4.7). RECALL en est absent : le modèle ne l'a jamais vu.
AFTER_HDR = frozenset({C.READ, C.SKIP})
AFTER_DEC_FULL = frozenset({C.CONT, C.STOP, C.NEXT})
GEN_CODES = frozenset({C.END, C.REFRESH, C.NOTE})
NOTE_CODES = frozenset({C.REFRESH})


def mask_for(allowed, device="cpu") -> torch.Tensor:
    """Masque additif (256,) : 0 sur les octets légaux, −inf ailleurs."""
    m = torch.full((256,), float("-inf"), dtype=torch.float32, device=device)
    m[torch.tensor(sorted(allowed), dtype=torch.long, device=device)] = 0.0
    return m


def masked_argmax(logits: torch.Tensor, allowed) -> int:
    return int((logits.float() + mask_for(allowed, logits.device)).argmax())
```

- [ ] **Step 4 : vérifier** — `python -m pytest tests/test_constrain.py -q` → 8 PASS ; puis `python -m pytest -q`.

- [ ] **Step 5 : commit** — `git add relis/infer/constrain.py tests/test_constrain.py && git commit -m "feat(infer): décodage contraint, automate UTF-8 complet et masques de phase"`

---

### Task 2 : le contrôleur

**Files:** Create `relis/infer/controller.py` ; Test `tests/test_controller.py`

**Interfaces:**
- Consumes : `constrain` (Task 1), `RelisModel.forward/step/new_state`, `State.reset_memory`, `relis.tape.tape.Segment`, `codes`.
- Produces :
  - `@dataclass Budget(max_read_bytes=2_000_000, max_segments=2_000, max_refresh=8, max_gen_bytes=4_096, chunk=4_096, max_note_bytes=200)`
  - `@dataclass Event(kind: str, channel: str = "", header: str = "", size: int = 0, action: str = "", byte: int = -1, text: str = "", index: int = 0, stats: dict = {})` — `kind` ∈ `{"turn_start", "channel", "segment", "decision", "scan_end", "gen_byte", "note", "refresh", "end", "turn_end"}`
  - `class Controller(model, device="cpu", budget=None)` avec `turn(query: bytes, history=(), docs=()) -> Iterator[Event]` et `answer_of(events) -> str` (fonction module).
  - `history` et `docs` sont des `Segment`, **du plus récent au plus ancien**, c'est la responsabilité de l'appelant.

- [ ] **Step 1 : tests**

`tests/test_controller.py` :
```python
import torch
from relis.model.config import RelisConfig
from relis.model.relis import RelisModel
from relis.tape import codes as C
from relis.tape.tape import Segment
from relis.tape.headers import format_header
from relis.infer.controller import Budget, Controller, Event


def _seg(role, txt, i=0):
    return Segment(header=format_header({"role": role, "i": i}), content=txt.encode("utf-8"))


def _ctl(**kw):
    m = RelisModel(RelisConfig.tiny()).eval()
    return Controller(m, "cpu", Budget(**kw))


def _kinds(events):
    return [e.kind for e in events]


def test_turn_without_history_reaches_generation():
    ev = list(_ctl(max_gen_bytes=12).turn(b"bonjour"))
    assert _kinds(ev)[0] == "turn_start" and _kinds(ev)[-1] == "turn_end"
    assert "channel" not in _kinds(ev)          # aucun canal ouvert
    assert ev[-1].stats["written"] <= 12


def test_scan_emits_one_segment_event_per_visited_segment():
    hist = [_seg("user", f"tour {i}", i) for i in range(4)]
    ev = list(_ctl(max_gen_bytes=8).turn(b"q", history=hist))
    segs = [e for e in ev if e.kind == "segment"]
    assert 1 <= len(segs) <= 4
    assert all(e.action in ("read", "skip") for e in segs)
    assert all(e.header for e in segs)


def test_last_history_segment_cannot_continue_without_docs():
    hist = [_seg("user", "a", 0)]
    ev = list(_ctl(max_gen_bytes=4).turn(b"q", history=hist))
    dec = [e.text for e in ev if e.kind == "decision"]
    assert dec == ["STOP"]                      # CONT et NEXT sont masqués


def test_last_history_segment_may_go_next_when_docs_exist():
    hist = [_seg("user", "a", 0)]
    docs = [Segment(header=format_header({"type": "doc", "name": "d.txt"}), content=b"x")]
    ev = list(_ctl(max_gen_bytes=4).turn(b"q", history=hist, docs=docs))
    dec = [e.text for e in ev if e.kind == "decision"]
    assert dec[0] in ("STOP", "NEXT")
    if dec[0] == "NEXT":
        assert [e.channel for e in ev if e.kind == "channel"] == ["history", "docs"]
        assert dec[-1] == "STOP"                # dernier document : seul STOP reste


def test_read_budget_forces_skip_then_stop():
    hist = [_seg("user", "x" * 400, i) for i in range(6)]
    ev = list(_ctl(max_read_bytes=10, max_gen_bytes=4).turn(b"q", history=hist))
    segs = [e for e in ev if e.kind == "segment"]
    assert segs[-1].action == "skip"
    assert [e for e in ev if e.kind == "scan_end"][0].text == "budget"


def test_generated_answer_is_valid_utf8_and_within_budget():
    ev = list(_ctl(max_gen_bytes=64).turn(b"bonjour"))
    raw = bytes(e.byte for e in ev if e.kind == "gen_byte")
    raw.decode("utf-8")                          # strict
    assert len(raw) <= 64
    assert ev[-1].stats["written"] == len(raw)


def test_stats_are_coherent():
    hist = [_seg("user", "abcdefghij", i) for i in range(3)]
    ev = list(_ctl(max_gen_bytes=8).turn(b"q", history=hist))
    st = ev[-1].stats
    assert st["available"] == 30
    assert 0 <= st["read"] <= 30 and 0.0 <= st["saved"] <= 1.0
    assert st["refresh"] <= 8 and st["seconds"] >= 0


def test_refresh_budget_is_respected(monkeypatch):
    """Un modèle truqué qui veut toujours REFRESH doit être arrêté par le budget."""
    ctl = _ctl(max_refresh=2, max_gen_bytes=200)
    real = ctl._decide

    def greedy(logits, allowed):
        if C.REFRESH in allowed:
            return C.REFRESH
        if C.END in allowed:
            return C.END
        return real(logits, allowed)

    ctl._decide = greedy
    ev = list(ctl.turn(b"q", history=[_seg("user", "a", 0)]))
    assert len([e for e in ev if e.kind == "refresh"]) == 2
    assert ev[-1].stats["refresh"] == 2


def test_refresh_resets_memory_but_keeps_slots():
    ctl = _ctl(max_refresh=1, max_gen_bytes=40)
    real = ctl._decide
    fired = {"n": 0}

    def once(logits, allowed):
        if C.REFRESH in allowed and fired["n"] == 0:
            fired["n"] = 1
            return C.REFRESH
        return real(logits, allowed)

    ctl._decide = once
    slots_before = []
    ev = []
    for e in ctl.turn(b"q", history=[_seg("user", "abc", 0)]):
        if e.kind == "refresh":
            slots_before.append(ctl.state.slots.clone())
            assert all(torch.count_nonzero(g[0]) == 0 for g in ctl.state.gdn if g is not None)
        ev.append(e)
    assert fired["n"] == 1 and slots_before
```

- [ ] **Step 2 : vérifier l'échec** — `python -m pytest tests/test_controller.py -q` → FAIL.

- [ ] **Step 3 : implémenter**

`relis/infer/controller.py` :
```python
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

        def encode(partial: bytes, note: bytes):
            self._feed(bytes([C.ENC]) + query, MODE_ENC)
            if partial:
                self._feed(bytes([C.PART]) + partial, MODE_ENC)
            if note:
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
                    self._feed(bytes([C.SEG]) + seg.header.encode("utf-8"), MODE_SCAN)
                    lg = self._feed(bytes([C.HDR]), MODE_SCAN)
                    a = self._decide(lg, frozenset({C.SKIP}) if over else AFTER_HDR)
                    self._feed_one(a, MODE_SCAN)
                    if a == C.READ:
                        content = C.sanitize(seg.content)
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
                if cnt["written"] >= bud.max_gen_bytes:
                    allowed = frozenset({C.END})
                else:
                    allowed = guard.allowed()
                    if guard.at_boundary:
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
                        while len(note) < bud.max_note_bytes:
                            na = ng.allowed() | (NOTE_CODES if ng.at_boundary else frozenset())
                            nb = self._decide(lg, na)
                            if nb == C.REFRESH:
                                break
                            note.append(nb)
                            ng.feed(nb)
                            lg = self._feed_one(nb, MODE_GEN)
                        yield Event("note", text=bytes(note).decode("utf-8", "replace"))
                    self._feed_one(C.REFRESH, MODE_GEN)
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

        encode(b"", b"")
        yield from scan()
        while True:
            yield from generate()
            if not pending["refresh"]:
                break
            self.state.reset_memory()
            encode(bytes(answer), pending["note"])
            yield from scan()

        cnt["seconds"] = round(time.time() - t0, 2)
        cnt["saved"] = round(1.0 - cnt["read"] / max(1, available), 3) if available else 0.0
        yield Event("turn_end", text=bytes(answer).decode("utf-8", "replace"), stats=dict(cnt))
```

- [ ] **Step 4 : vérifier** — `python -m pytest tests/test_controller.py -q` → 9 PASS ; puis `python -m pytest -q`. Si `test_refresh_resets_memory_but_keeps_slots` échoue, vérifier que `reset_memory` est bien appelé **après** avoir avalé REFRESH et **avant** le ré-encodage.

- [ ] **Step 5 : commit** — `git add relis/infer/controller.py tests/test_controller.py && git commit -m "feat(infer): contrôleur, budgets, cas limites câblés et flux d'événements"`

---

### Task 3 : rendu terminal et CLI

**Files:** Create `relis/infer/render.py`, `relis/infer/cli.py` ; Test `tests/test_render.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes : `Event`, `Controller`, `Budget`, `relis.infer.sample.load_model`, `relis.train.loop.fetch_weights_path`, `relis.tape.headers.format_header`.
- Produces :
  - `render.ANSI` (dict de codes), `render.supports_color(stream) -> bool`, `class TerminalRenderer(stream=sys.stdout, color=None, quiet=False)` avec `handle(event)` et `run(events) -> str` (renvoie la réponse).
  - `cli.load_docs(paths) -> list[Segment]`, `cli.history_segments(turns) -> list[Segment]` (du plus récent au plus ancien), `cli.main(argv=None)`.
  - Fichier de conversation JSON : `{"turns": [{"role": "user"|"assistant", "text": str, "t": str}]}`.

- [ ] **Step 1 : tests**

`tests/test_render.py` :
```python
import io
from relis.infer.controller import Event
from relis.infer.render import TerminalRenderer, supports_color


def _events():
    return [
        Event("turn_start", text="Quelle est l'adresse ?"),
        Event("channel", channel="history", size=3),
        Event("segment", channel="history", header="role=user;i=0", size=12, action="read"),
        Event("decision", channel="history", text="CONT"),
        Event("segment", channel="history", header="role=user;i=1", size=40, action="skip"),
        Event("decision", channel="history", text="STOP"),
        Event("scan_end", text="stop"),
        *[Event("gen_byte", byte=b) for b in "ok.".encode()],
        Event("end"),
        Event("turn_end", text="ok.", stats={"read": 12, "written": 3, "refresh": 0,
                                             "segments": 2, "available": 52,
                                             "seconds": 0.4, "saved": 0.769}),
    ]


def test_render_shows_segments_decisions_and_counts():
    buf = io.StringIO()
    out = TerminalRenderer(buf, color=False).run(_events())
    s = buf.getvalue()
    assert "role=user;i=0" in s and "role=user;i=1" in s
    assert "lu" in s and "sauté" in s and "STOP" in s
    assert "12" in s and "77" in s              # octets lus et % économisé
    assert out == "ok."


def test_render_without_color_emits_no_escape():
    buf = io.StringIO()
    TerminalRenderer(buf, color=False).run(_events())
    assert "\x1b[" not in buf.getvalue()


def test_render_with_color_emits_escape():
    buf = io.StringIO()
    TerminalRenderer(buf, color=True).run(_events())
    assert "\x1b[" in buf.getvalue()


def test_quiet_prints_only_the_answer():
    buf = io.StringIO()
    TerminalRenderer(buf, color=False, quiet=True).run(_events())
    assert buf.getvalue().strip() == "ok."


def test_note_and_refresh_are_shown():
    buf = io.StringIO()
    ev = [Event("note", text="il manque la ville"), Event("refresh", index=1),
          Event("turn_end", text="", stats={"read": 0, "written": 0, "refresh": 1,
                                            "segments": 0, "available": 0,
                                            "seconds": 0.1, "saved": 0.0})]
    TerminalRenderer(buf, color=False).run(ev)
    assert "il manque la ville" in buf.getvalue()
    assert "relecture" in buf.getvalue()


def test_supports_color_on_plain_stringio():
    assert supports_color(io.StringIO()) is False
```

`tests/test_cli.py` :
```python
import json
import os
import torch
from relis.model.config import RelisConfig
from relis.model.relis import RelisModel
from relis.train.checkpoint import save_checkpoint
from relis.infer import cli


def _ckpt(tmp_path):
    cfg = RelisConfig.tiny()
    save_checkpoint(str(tmp_path), RelisModel(cfg), None, None, step=7,
                    cfg_dict=cfg.to_dict(), extra={})
    return str(tmp_path / "last.pt")


def test_load_docs_builds_segments_newest_first(tmp_path):
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.md").write_text("bravo", encoding="utf-8")
    segs = cli.load_docs([str(tmp_path / "a.txt"), str(tmp_path / "b.md")])
    assert [s.content for s in segs] == [b"bravo", b"alpha"]     # dernier joint en premier
    assert "name=b.md" in segs[0].header and "type=doc" in segs[0].header


def test_history_segments_are_newest_first():
    turns = [{"role": "user", "text": "un"}, {"role": "assistant", "text": "deux"},
             {"role": "user", "text": "trois"}]
    segs = cli.history_segments(turns)
    assert [s.content for s in segs] == [b"trois", b"deux", b"un"]
    assert "role=user" in segs[0].header


def test_cli_single_question_writes_conversation(tmp_path, capsys):
    conv = tmp_path / "conv.json"
    cli.main(["--ckpt", _ckpt(tmp_path), "--ask", "bonjour",
              "--conversation", str(conv), "--max_gen", "8", "--no-color"])
    data = json.loads(conv.read_text(encoding="utf-8"))
    assert [t["role"] for t in data["turns"]] == ["user", "assistant"]
    assert data["turns"][0]["text"] == "bonjour"
    assert capsys.readouterr().out.strip() != ""


def test_cli_second_turn_reads_previous_history(tmp_path, capsys):
    conv = tmp_path / "conv.json"
    ck = _ckpt(tmp_path)
    cli.main(["--ckpt", ck, "--ask", "un", "--conversation", str(conv), "--max_gen", "6", "--no-color"])
    cli.main(["--ckpt", ck, "--ask", "deux", "--conversation", str(conv), "--max_gen", "6", "--no-color"])
    data = json.loads(conv.read_text(encoding="utf-8"))
    assert len(data["turns"]) == 4
    assert "history" in capsys.readouterr().out         # le canal a été ouvert au 2e tour


def test_cli_quiet_prints_only_answer(tmp_path, capsys):
    cli.main(["--ckpt", _ckpt(tmp_path), "--ask", "salut", "--max_gen", "6", "--quiet", "--no-color"])
    out = capsys.readouterr().out
    assert "octets lus" not in out
```

- [ ] **Step 2 : vérifier l'échec** — `python -m pytest tests/test_render.py tests/test_cli.py -q` → FAIL.

- [ ] **Step 3 : implémenter le rendu**

`relis/infer/render.py` :
```python
"""Rendu terminal du flux d'événements (spec §8) : le terminal est la démonstration.

Couleurs en ANSI direct, sans dépendance. Repli automatique quand la sortie n'est pas
un terminal, et option explicite pour forcer l'un ou l'autre.
"""
import os
import sys

ANSI = {"reset": "\x1b[0m", "dim": "\x1b[2m", "bold": "\x1b[1m",
        "teal": "\x1b[36m", "amber": "\x1b[33m", "green": "\x1b[32m", "grey": "\x1b[90m"}


def supports_color(stream) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


class TerminalRenderer:
    def __init__(self, stream=None, color=None, quiet: bool = False):
        self.out = stream or sys.stdout
        self.color = supports_color(self.out) if color is None else bool(color)
        self.quiet = quiet
        self._answer = []

    def _c(self, name: str, s: str) -> str:
        return f"{ANSI[name]}{s}{ANSI['reset']}" if self.color else s

    def _w(self, s: str = "") -> None:
        if not self.quiet:
            self.out.write(s + "\n")

    def handle(self, e) -> None:
        k = e.kind
        if k == "turn_start":
            self._w()
            self._w(self._c("bold", "? ") + e.text)
        elif k == "channel":
            self._w(self._c("teal", f"  ⟨{e.channel}⟩ ") + self._c("grey", f"{e.size} segments"))
        elif k == "segment":
            lu = e.action == "read"
            mark = self._c("green", "  ✓") if lu else self._c("grey", "  ⨯")
            size = f"{e.size:>7} o" if lu else f"{'—':>7}  "
            etat = "lu" if lu else "sauté"
            self._w(f"{mark} {e.header[:56]:<56} {self._c('grey', size)}  {self._c('grey', etat)}")
        elif k == "decision" and e.text in ("STOP", "NEXT"):
            self._w(self._c("amber", f"     ■ {e.text}"))
        elif k == "note":
            self._w(self._c("amber", "     ✎ ") + e.text)
        elif k == "refresh":
            self._w(self._c("amber", f"     ↻ relecture {e.index}"))
        elif k == "gen_byte":
            ch = bytes([e.byte])
            self._answer.append(ch)
            if not self.quiet:
                self.out.write(ch.decode("utf-8", "replace"))
                self.out.flush()
        elif k == "end":
            self._w()
        elif k == "turn_end":
            s = e.stats
            if self.quiet:
                self.out.write(e.text + "\n")
            else:
                self._w(self._c("grey",
                                f"  {s['read']} octets lus · {s['written']} écrits · "
                                f"{s['refresh']} relecture(s) · {int(s['saved'] * 100)} % "
                                f"de l'historique économisé · {s['seconds']} s"))

    def run(self, events) -> str:
        for e in events:
            self.handle(e)
        return b"".join(self._answer).decode("utf-8", "replace")
```

- [ ] **Step 4 : implémenter la CLI**

`relis/infer/cli.py` :
```python
"""Conversation en ligne de commande (spec §8).

    python -m relis.infer.cli --repo jaafar2022/relis-v1-tape --ask "Quelle est l'adresse ?"
    python -m relis.infer.cli --ckpt runs/tape1/last.pt --doc notes.md --doc api.py \\
        --conversation conv.json --ask "Où est le serveur ?"
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
    return [Segment(header=format_header({"role": t["role"], "t": t.get("t", "")}),
                    content=t["text"].encode("utf-8"))
            for t in reversed(turns)]


def _load_conversation(path):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("turns", [])
    return []


def _save_conversation(path, turns):
    if not path:
        return
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
```

- [ ] **Step 5 : vérifier** — `python -m pytest tests/test_render.py tests/test_cli.py -q` → 10 PASS ; puis `python -m pytest -q`.

- [ ] **Step 6 : essai manuel sur CPU**

```bash
python - <<'PY'
from relis.model.config import RelisConfig
from relis.model.relis import RelisModel
from relis.train.checkpoint import save_checkpoint
cfg = RelisConfig.tiny()
save_checkpoint("runs/cli_smoke", RelisModel(cfg), None, None, 0, cfg.to_dict(), {})
PY
printf 'alpha bravo charlie\n' > /tmp/doc_a.txt
python -m relis.infer.cli --ckpt runs/cli_smoke/last.pt --doc /tmp/doc_a.txt \
    --ask "bonjour" --max_gen 40 --no-color
```
Expected : l'en-tête du document s'affiche avec son sort, une ligne STOP, du texte incohérent (modèle minuscule non entraîné) puis la ligne de comptes. Aucune exception.

- [ ] **Step 7 : commit** — `git add relis/infer/render.py relis/infer/cli.py tests/test_render.py tests/test_cli.py && git commit -m "feat(infer): rendu terminal du flux d'événements et CLI conversationnelle"`

---

### Task 4 : exposer les épisodes complets et le harnais d'évaluation

**Files:** Modify `relis/data/episodes.py` ; Create `relis/eval/__init__.py`, `relis/eval/harness.py` ; Test `tests/test_cases.py`

**Pourquoi ce refactor.** `generate_episode` renvoie un `TapeSpec` dont les passes ne contiennent **que les segments lus par l'oracle** : l'historique complet est perdu. Pour évaluer en autonomie il faut donner au contrôleur l'historique entier et laisser le modèle décider où s'arrêter. On extrait donc la matière première dans un `Case`, et le `TapeSpec` en devient une fonction.

**Interfaces:**
- Produces dans `episodes.py` :
  - `@dataclass Case(kind, query: bytes, history: list, docs: list, doc_relevant: list, passes_needed: list, answer_parts: list, notes: list, answer_value: str, stop_index: int)` où `history` et `docs` sont **complets**, du plus récent au plus ancien ; `passes_needed` est une liste de `(needed_hist: set[int], needed_doc: int | None)` ; `stop_index` est l'indice du segment d'historique où l'oracle s'arrête à la première passe, ou `-1` si l'arrêt a lieu dans les documents ; `answer_value` est la chaîne qui doit figurer dans la réponse (vide pour `absent`).
  - `case_to_spec(case) -> TapeSpec` (applique `oracle_pass` passe par passe).
  - `generate_case(rng, kind=None) -> Case` ; `iter_cases(seed, n) -> Iterator[Case]`.
  - `generate_episode(rng, kind=None)` reste inchangé du dehors : il appelle `generate_case` puis `case_to_spec`, garde de taille comprise.
- Produces dans `eval/harness.py` :
  - `load_controller(ckpt_or_repo, device, budget=None) -> (Controller, step)`
  - `run_case(ctl, case) -> dict` : `{"answer", "read", "written", "refresh", "saved", "seconds", "stop_at", "read_segments", "skipped_docs", "events"}` où `stop_at` est l'indice du segment d'historique sur lequel le contrôleur a émis STOP (ou `-1` s'il a arrêté dans les documents, `None` s'il n'a jamais émis STOP).
  - `judge(case, answer) -> bool` : pour `absent`, vrai si la réponse contient « ne trouve pas » ou « ne sais pas » ; sinon vrai si `case.answer_value` apparaît dans la réponse, comparaison insensible à la casse et aux espaces multiples.

- [ ] **Step 1 : tests**

`tests/test_cases.py` :
```python
import random
from relis.tape import codes as C
from relis.tape.tape import build_tape, validate_spec
from relis.data.episodes import (KINDS, Case, case_to_spec, generate_case, generate_episode,
                                 iter_cases)


def test_case_history_is_complete_and_spec_is_truncated():
    c = generate_case(random.Random(4), "fact_recall")
    spec = case_to_spec(c)
    assert len(spec.passes[0].history) <= len(c.history)
    assert spec.passes[0].history[-1].after == C.STOP
    assert c.stop_index == len(spec.passes[0].history) - 1


def test_every_kind_produces_a_valid_case_and_spec():
    rng = random.Random(0)
    for kind in KINDS:
        for _ in range(5):
            c = generate_case(rng, kind)
            assert isinstance(c, Case) and c.query and c.history is not None
            assert len(c.passes_needed) == len(c.answer_parts)
            assert len(c.notes) == len(c.answer_parts) - 1
            spec = case_to_spec(c)
            validate_spec(spec)
            assert build_tape(spec).decisions()[-1] == C.END


def test_answer_value_appears_in_the_answer_when_expected():
    rng = random.Random(9)
    for kind in ("fact_recall", "what_did_i_say", "doc_lookup", "variable_tracking"):
        c = generate_case(rng, kind)
        joined = b"".join(c.answer_parts).decode("utf-8", "replace")
        assert c.answer_value and c.answer_value in joined, kind


def test_absent_has_no_answer_value():
    c = generate_case(random.Random(2), "absent")
    assert c.answer_value == ""
    assert c.passes_needed == [(set(), None)]


def test_generate_episode_still_deterministic_and_bounded():
    a = [bytes(build_tape(s).data) for s in (generate_episode(random.Random(7)) for _ in range(20))]
    b = [bytes(build_tape(s).data) for s in (generate_episode(random.Random(7)) for _ in range(20))]
    assert a == b
    assert all(len(t) <= 15_000 for t in a)


def test_iter_cases_is_deterministic():
    x = [c.query for c in iter_cases(3, 15)]
    y = [c.query for c in iter_cases(3, 15)]
    assert x == y
```

Et dans `tests/test_eval.py` (créé ici, complété à la Task 5) :
```python
import random
import torch
from relis.model.config import RelisConfig
from relis.model.relis import RelisModel
from relis.train.checkpoint import save_checkpoint
from relis.data.episodes import generate_case
from relis.infer.controller import Budget, Controller
from relis.eval.harness import judge, load_controller, run_case


def _ctl():
    return Controller(RelisModel(RelisConfig.tiny()).eval(), "cpu",
                      Budget(max_gen_bytes=24, max_read_bytes=4000))


def test_run_case_reports_counters_and_stop():
    c = generate_case(random.Random(1), "fact_recall")
    r = run_case(_ctl(), c)
    assert set(r) >= {"answer", "read", "written", "saved", "stop_at", "read_segments"}
    assert r["written"] <= 24
    assert r["stop_at"] is None or isinstance(r["stop_at"], int)


def test_judge_absent_and_value():
    c = generate_case(random.Random(2), "absent")
    assert judge(c, "Je ne trouve pas cette information.") is True
    assert judge(c, "C'est 42.") is False
    f = generate_case(random.Random(3), "fact_recall")
    assert judge(f, f"La réponse est {f.answer_value} voilà") is True
    assert judge(f, "aucune idée") is False


def test_load_controller_from_local_checkpoint(tmp_path):
    cfg = RelisConfig.tiny()
    save_checkpoint(str(tmp_path), RelisModel(cfg), None, None, 11, cfg.to_dict(), {})
    ctl, step = load_controller(str(tmp_path / "last.pt"), "cpu")
    assert step == 11 and isinstance(ctl, Controller)
```

- [ ] **Step 2 : vérifier l'échec** — `python -m pytest tests/test_cases.py tests/test_eval.py -q` → FAIL.

- [ ] **Step 3 : refactoriser `episodes.py`**

Ajouter en tête du module (après les imports) :
```python
from dataclasses import dataclass, field


@dataclass
class Case:
    """Matière première d'un épisode : l'historique et les documents **complets**,
    plus ce que l'oracle sait. `case_to_spec` en tire le ruban tronqué au STOP ;
    l'évaluation en autonomie, elle, donne l'historique entier au contrôleur."""
    kind: str
    query: bytes
    history: list                  # Segment, récent → ancien, complet
    docs: list                     # Segment, complet
    doc_relevant: list
    passes_needed: list            # [(needed_hist:set[int], needed_doc:int|None), ...]
    answer_parts: list
    notes: list
    answer_value: str = ""
    stop_index: int = -1


def case_to_spec(case: Case) -> TapeSpec:
    passes = [oracle_pass(case.history, nh, case.docs, nd, case.doc_relevant)
              for nh, nd in case.passes_needed]
    first = passes[0]
    case.stop_index = len(first.history) - 1 if first.docs is None else -1
    return TapeSpec(query=case.query, passes=passes,
                    answer_parts=case.answer_parts, notes=case.notes)
```

Puis transformer chacune des sept fonctions `_kind` : elle construit désormais un `Case` au lieu d'un `TapeSpec`, **sans changer l'ordre de consommation du générateur aléatoire**. Exemple pour `_fact_recall` — appliquer le même patron aux six autres :
```python
def _fact_recall(rng) -> Case:
    f = rng.choice(FACTS); v = f[4](rng)
    n = rng.randint(3, 14); at = rng.randint(0, n - 1)
    hist = _history(rng, n, {at: f[0].format(v=v)})
    idx = 2 * (n - 1 - at) + 1
    return Case(kind="fact_recall", query=f[1].encode("utf-8"), history=hist, docs=[],
                doc_relevant=[], passes_needed=[({idx}, None)],
                answer_parts=[f[2].format(v=v).encode("utf-8")], notes=[], answer_value=str(v))
```
Correspondances pour les autres : `_variable_tracking` → `answer_value=str(val)`, `passes_needed=[(needed, None)]` ; `_what_did_i_say` → `answer_value=v` ; `_doc_lookup` → `docs=docs, doc_relevant=relevant, passes_needed=[(set(), target)]`, `answer_value=str(v)` ; `_absent` → `docs=docs, doc_relevant=[False]*n_docs, passes_needed=[(set(), None)]`, `answer_value=""` ; `_two_facts` → `passes_needed=[({ia}, None), ({ia, ib}, None)]`, `answer_value=str(vb)` ; `_long_answer` → `passes_needed=[(needed, None)] * len(parts)`, `answer_value=str(v)`.

Renommer le point d'entrée et garder l'ancien :
```python
def generate_case(rng: random.Random, kind: str | None = None) -> Case:
    if kind is None:
        kind = rng.choices(KINDS, weights=KIND_WEIGHTS, k=1)[0]
    for _ in range(MAX_REDRAWS + 1):
        case = _GEN[kind](rng)
        spec = case_to_spec(case)
        if len(build_tape(spec)) <= MAX_TAPE_BYTES:
            return case
    raise RuntimeError(f"épisode « {kind} » toujours au-dessus de {MAX_TAPE_BYTES} octets")


def generate_episode(rng: random.Random, kind: str | None = None) -> TapeSpec:
    return case_to_spec(generate_case(rng, kind))


def iter_cases(seed: int, n: int):
    rng = random.Random(seed)
    for _ in range(n):
        yield generate_case(rng)
```
`iter_episodes` reste tel quel mais délègue : `yield case_to_spec(generate_case(rng))`.

- [ ] **Step 4 : implémenter le harnais**

`relis/eval/__init__.py` : vide.

`relis/eval/harness.py` :
```python
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
            "skipped_docs": sum(1 for e in doc_segments if e.action == "skip"), "events": events}
```

- [ ] **Step 5 : vérifier** — `python -m pytest tests/test_cases.py tests/test_eval.py tests/test_episodes.py tests/test_pack.py -q` → PASS. **Les tests d'épisodes du plan 2a doivent rester verts sans modification** : c'est le filet du refactor. Puis `python -m pytest -q`.

- [ ] **Step 6 : commit** — `git add relis/data/episodes.py relis/eval tests/test_cases.py tests/test_eval.py && git commit -m "feat(eval): exposer les épisodes complets en Case et harnais d'évaluation"`

---

### Task 5 : décisions en autonomie

**Files:** Create `relis/eval/autonomy.py` ; Modify `tests/test_eval.py`

**Interfaces:**
- Produces : `classify(case, result) -> str` ∈ `{"exact", "trop_tot", "trop_tard", "doc_utile_saute", "jamais_stop"}` ; `evaluate(ctl, cases) -> dict` renvoyant les comptes par classe, le taux de réponses jugées correctes, les octets lus moyens et la part d'historique économisée ; `format_report(res) -> str` ; CLI `python -m relis.eval.autonomy --repo … --n 200 --seed 777`.

Règles de classification, pour un cas dont l'arrêt attendu est `case.stop_index` dans l'historique :
- `jamais_stop` si le contrôleur n'a jamais émis STOP ;
- `doc_utile_saute` si le document nécessaire existe et a été sauté ;
- `exact` si `stop_at == case.stop_index` ;
- `trop_tot` si `stop_at < case.stop_index` (le fait n'a pas été lu) ;
- `trop_tard` sinon (lecture inutile, réponse en principe correcte).

- [ ] **Step 1 : tests** — ajouter à `tests/test_eval.py` :
```python
from relis.data.episodes import iter_cases
from relis.eval.autonomy import classify, evaluate, format_report


def test_classify_covers_every_case():
    c = generate_case(random.Random(5), "fact_recall")
    c.stop_index = 3
    assert classify(c, {"stop_at": 3, "skipped_docs": 0}) == "exact"
    assert classify(c, {"stop_at": 1, "skipped_docs": 0}) == "trop_tot"
    assert classify(c, {"stop_at": 6, "skipped_docs": 0}) == "trop_tard"
    assert classify(c, {"stop_at": None, "skipped_docs": 0}) == "jamais_stop"
    d = generate_case(random.Random(6), "doc_lookup")
    assert classify(d, {"stop_at": -1, "skipped_docs": 99}) == "doc_utile_saute"


def test_evaluate_returns_shares_that_sum_to_one():
    res = evaluate(_ctl(), list(iter_cases(11, 6)))
    assert abs(sum(res["classes"].values()) - 1.0) < 1e-6
    assert 0.0 <= res["exactitude"] <= 1.0
    assert res["n"] == 6 and res["octets_lus_moyen"] >= 0
    assert "exact" in format_report(res)
```

- [ ] **Step 2 : vérifier l'échec.**

- [ ] **Step 3 : implémenter**

`relis/eval/autonomy.py` :
```python
"""Décisions en autonomie contre l'oracle (spec §9).

En forçage, chaque décision est jugée en supposant correctes celles qui la précèdent.
Ici le contrôleur décide seul : une erreur de STOP change tout ce qu'il voit ensuite.
C'est la seule mesure qui dit si le mécanisme survit à la composition de ses erreurs.
"""
import argparse
from collections import Counter

from relis.data.episodes import iter_cases
from .harness import judge, load_controller, run_case

CLASSES = ("exact", "trop_tot", "trop_tard", "doc_utile_saute", "jamais_stop")


def classify(case, result) -> str:
    if result["stop_at"] is None:
        return "jamais_stop"
    needed_doc = case.passes_needed[0][1]
    if needed_doc is not None and result["skipped_docs"] and result["stop_at"] == -1:
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
```

- [ ] **Step 4 : vérifier** puis **Step 5 : commit** — `git add relis/eval/autonomy.py tests/test_eval.py && git commit -m "feat(eval): décisions en autonomie, classées contre l'oracle"`

---

### Task 6 : aiguille, calcul adaptatif et mode d'emploi

**Files:** Create `relis/eval/needle.py`, `relis/eval/adaptive.py`, `notebooks/COLAB_EVAL.md` ; Modify `tests/test_eval.py`, `docs/CARNET.md`

**Interfaces:**
- `needle.build_case(rng, target_bytes) -> Case` : un `fact_recall` dont l'historique est rallongé par des tours de remplissage jusqu'à atteindre environ `target_bytes`, le fait restant à une position tirée au hasard. `needle.evaluate(ctl, sizes, n_per_size, seed) -> list[dict]` avec par taille : `taille`, `exactitude`, `octets_lus_moyen`, `secondes_moyennes`. `format_table(rows) -> str`. CLI `python -m relis.eval.needle --repo … --sizes 1000,4000,16000,64000 --n 20`.
- `adaptive.evaluate(ctl, n, seed) -> dict` : compare les octets lus sur des cas faciles (`fact_recall` dont le fait est dans les trois segments les plus récents) et difficiles (fait dans le tiers le plus ancien, ou `doc_lookup` à plusieurs documents), et renvoie `{"faciles", "difficiles", "rapport", "exactitude_faciles", "exactitude_difficiles"}`. CLI analogue.

- [ ] **Step 1 : tests** — ajouter à `tests/test_eval.py` :
```python
from relis.eval.needle import build_case, evaluate as needle_eval, format_table
from relis.eval.adaptive import evaluate as adaptive_eval


def test_needle_case_reaches_requested_size_and_keeps_the_fact():
    c = build_case(random.Random(1), 6000)
    total = sum(len(s.content) for s in c.history)
    assert 4000 <= total <= 12000
    joined = b" ".join(s.content for s in c.history).decode("utf-8", "replace")
    assert c.answer_value in joined
    assert 0 <= c.stop_index < len(c.history)


def test_needle_evaluate_returns_one_row_per_size():
    rows = needle_eval(_ctl(), sizes=(600, 1200), n_per_size=2, seed=5)
    assert [r["taille"] for r in rows] == [600, 1200]
    assert all(0.0 <= r["exactitude"] <= 1.0 for r in rows)
    assert "taille" in format_table(rows)


def test_adaptive_returns_two_populations_and_a_ratio():
    res = adaptive_eval(_ctl(), n=3, seed=5)
    assert res["faciles"] >= 0 and res["difficiles"] >= 0
    assert res["rapport"] > 0
```

- [ ] **Step 2 : vérifier l'échec.**

- [ ] **Step 3 : implémenter `needle.py`**

```python
"""Aiguille : exactitude et coût selon la longueur de l'historique (spec §9).

Le fait est placé dans un historique qu'on rallonge jusqu'à la taille visée, bien
au-delà des longueurs vues à l'entraînement. Un Transformer de même taille s'effondre
dès que l'historique dépasse sa fenêtre ; RELIS n'a pas de fenêtre, la question est
donc de savoir si sa Mémoire tient.
"""
import argparse
import random

from relis.data.episodes import FACTS, Case, _history, case_to_spec
from .harness import judge, load_controller, run_case


def build_case(rng: random.Random, target_bytes: int) -> Case:
    f = rng.choice(FACTS)
    v = f[4](rng)
    n = 4
    while True:
        at = rng.randint(0, n - 1)
        hist = _history(rng, n, {at: f[0].format(v=v)})
        if sum(len(s.content) for s in hist) >= target_bytes or n > 4000:
            break
        n = max(n + 2, int(n * 1.6))
    idx = 2 * (n - 1 - at) + 1
    case = Case(kind="needle", query=f[1].encode("utf-8"), history=hist, docs=[],
                doc_relevant=[], passes_needed=[({idx}, None)],
                answer_parts=[f[2].format(v=v).encode("utf-8")], notes=[], answer_value=str(v))
    case_to_spec(case)                     # renseigne stop_index
    return case


def evaluate(ctl, sizes=(1_000, 4_000, 16_000, 64_000, 200_000), n_per_size: int = 20, seed: int = 777):
    rows = []
    for size in sizes:
        rng = random.Random(seed + size)
        bons, lus, secs = 0, 0, 0.0
        for _ in range(n_per_size):
            case = build_case(rng, size)
            r = run_case(ctl, case)
            bons += int(judge(case, r["answer"]))
            lus += r["read"]; secs += r["seconds"]
        rows.append({"taille": size, "exactitude": bons / n_per_size,
                     "octets_lus_moyen": lus // n_per_size,
                     "secondes_moyennes": round(secs / n_per_size, 2)})
    return rows


def format_table(rows) -> str:
    out = [f"{'taille':>9} {'exactitude':>11} {'octets lus':>11} {'secondes':>9}"]
    for r in rows:
        out.append(f"{r['taille']:>9} {r['exactitude']:>11.3f} "
                   f"{r['octets_lus_moyen']:>11} {r['secondes_moyennes']:>9}")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo"); ap.add_argument("--ckpt")
    ap.add_argument("--sizes", default="1000,4000,16000,64000")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ctl, step = load_controller(args.ckpt or args.repo, device)
    print(f"[aiguille] checkpoint au pas {step}")
    print(format_table(evaluate(ctl, tuple(int(s) for s in args.sizes.split(",")), args.n, args.seed)))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4 : implémenter `adaptive.py`**

```python
"""Calcul adaptatif : le coût suit-il la difficulté ? (spec §9)

Facile = le fait est dans les trois segments les plus récents, le modèle devrait
s'arrêter presque tout de suite. Difficile = le fait est dans le tiers le plus ancien,
ou la réponse est dans un document parmi plusieurs. On compare les octets lus.
"""
import argparse
import random

from relis.data.episodes import FACTS, Case, _history, case_to_spec, generate_case
from .harness import judge, load_controller, run_case


def _placed_case(rng: random.Random, recent: bool) -> Case:
    f = rng.choice(FACTS); v = f[4](rng)
    n = rng.randint(8, 14)
    at = rng.randint(n - 3, n - 1) if recent else rng.randint(0, max(0, n // 3))
    hist = _history(rng, n, {at: f[0].format(v=v)})
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo"); ap.add_argument("--ckpt")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ctl, step = load_controller(args.ckpt or args.repo, device)
    print(f"[adaptatif] checkpoint au pas {step}")
    print(format_report(evaluate(ctl, args.n, args.seed)))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5 : écrire `notebooks/COLAB_EVAL.md`**

Un mode d'emploi en trois cellules Colab : installation avec `pip install -q -e . --no-deps`, jeton dans les secrets, puis les trois commandes `python -m relis.eval.autonomy`, `python -m relis.eval.needle`, `python -m relis.eval.adaptive` sur `--repo jaafar2022/relis-v1-tape`, plus un exemple de conversation avec `python -m relis.infer.cli`. Expliquer comment lire chaque sortie : ce que signifient `trop_tot` et `trop_tard`, pourquoi l'exactitude de l'aiguille doit rester plate quand la taille croît, et pourquoi le rapport du calcul adaptatif doit être nettement supérieur à 1.

- [ ] **Step 6 : mettre à jour `docs/CARNET.md`** — passer le plan 3 de « à faire » à « fait », y porter les résultats réels des trois évaluations une fois exécutées sur GPU, et retirer les Extraits du périmètre du plan 3 en les renvoyant à un plan ultérieur.

- [ ] **Step 7 : vérifier** — `python -m pytest -q` (durée à reporter) puis essai manuel :
```bash
python -m relis.eval.autonomy --ckpt runs/cli_smoke/last.pt --n 4 --device cpu
python -m relis.eval.needle --ckpt runs/cli_smoke/last.pt --sizes 600,1200 --n 2 --device cpu
python -m relis.eval.adaptive --ckpt runs/cli_smoke/last.pt --n 2 --device cpu
```
Expected : trois rapports formatés, chiffres médiocres (modèle minuscule non entraîné), aucune exception.

- [ ] **Step 8 : commit** — `git add relis/eval/needle.py relis/eval/adaptive.py notebooks/COLAB_EVAL.md tests/test_eval.py docs/CARNET.md && git commit -m "feat(eval): aiguille selon la longueur, calcul adaptatif, mode d'emploi Colab"`

---

## Auto-revue du plan

**Couverture de la spec.** §7.1 contrôleur, primitive unique, cas limites câblés, budgets, RECALL exclu : Task 2. §7.2 décodage contraint et automate UTF-8 complet : Task 1. §7.3 balayage en mode bloc et génération pas-à-pas : Task 2 (`_feed` contre `_feed_one`). §4.8 REFRESH à l'inférence par `reset_memory` puis ré-encodage avec PART et NOTE : Task 2. §8 le terminal comme démonstration, documents joints, conversation persistante : Task 3. §9 autonomie, aiguille selon la longueur, calcul adaptatif, zéro caractère invalide : Tasks 1, 5, 6. Hors périmètre assumé : Extraits et RECALL (exigent de régénérer les rubans et de réentraîner), sonde du Buffer (dépend de l'étiquetage du professeur, plan 2b), application Gradio (après la CLI).

**Placeholders.** Aucun ; chaque étape porte son code. Les essais manuels sur CPU utilisent un modèle minuscule non entraîné et n'attendent que l'absence d'exception ; les chiffres réels viendront du checkpoint Kaggle.

**Cohérence des types.** `Segment(header: str, content: bytes)` circule de `cli.load_docs` et `cli.history_segments` vers `Controller.turn`, et de `Case.history` vers `run_case` ; l'ordre récent → ancien est la responsabilité de l'appelant, documenté aux trois endroits. `Event` est produit par `Controller.turn` et consommé par `TerminalRenderer.handle`, `answer_of` et `harness.run_case` : le champ `channel` des événements `decision` est ce qui permet à `run_case` de distinguer un STOP dans l'historique d'un STOP dans les documents. `frozenset[int]` est le type des ensembles d'octets autorisés, de `constrain` jusqu'à `Controller._decide`. `Case.stop_index` est renseigné par `case_to_spec`, donc `build_case` de `needle` et `_placed_case` d'`adaptive` appellent `case_to_spec` avant de renvoyer le cas.
