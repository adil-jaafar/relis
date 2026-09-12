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
