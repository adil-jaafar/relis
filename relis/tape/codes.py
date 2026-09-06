"""Codes de contrôle du ruban (spec §4.2) et adresses de cellules.

Le vocabulaire est exactement 256 octets. Les octets 0x00-0x1F, sauf
\t, \n, \r, sont réservés comme codes de contrôle.
"""
from enum import IntEnum

# Marqueurs (insérés par le contrôleur)
ENC = 0x01
SEG = 0x02
SCAN = 0x0E
GEN = 0x0F
DEC = 0x10
PART = 0x12
CHAN = 0x1C
HDR = 0x1F
# Actions (prédites par le modèle)
END = 0x03
STOP = 0x04
REFRESH = 0x05
NEXT = 0x06
CONT = 0x07
SKIP = 0x08
READ = 0x11
RECALL = 0x13
NOTE = 0x14

TEXT_CONTROL = frozenset({0x09, 0x0A, 0x0D})
CONTROL_CODES = frozenset(b for b in range(0x20) if b not in TEXT_CONTROL)

DECISION_CODES = frozenset({READ, SKIP, CONT, STOP, NEXT, END, REFRESH, NOTE})

_NAMES = {ENC: "ENC", SEG: "SEG", END: "END", STOP: "STOP", REFRESH: "REFRESH", NEXT: "NEXT",
          CONT: "CONT", SKIP: "SKIP", SCAN: "SCAN", GEN: "GEN", DEC: "DEC", READ: "READ",
          PART: "PART", RECALL: "RECALL", NOTE: "NOTE", CHAN: "CHAN", HDR: "HDR"}


def name(b: int) -> str:
    return _NAMES.get(b, f"0x{b:02x}")


REPLACEMENT = b"\xef\xbf\xbd"  # U+FFFD


class Mode(IntEnum):
    ENCODE = 0
    SCAN = 1
    GENERATE = 2


_SANITIZE_TABLE = {b: REPLACEMENT for b in CONTROL_CODES}


def sanitize(data: bytes) -> bytes:
    """Remplace chaque octet de contrôle d'un texte d'entrée par U+FFFD."""
    if not any(b in CONTROL_CODES for b in data):
        return bytes(data)
    return b"".join(_SANITIZE_TABLE.get(b, bytes((b,))) for b in data)


# Adresses de cellules (spec §4.2) : octets dans 0x20-0xFF.
ADDR_BASE = 0x20
ADDR_RANGE = 0x100 - ADDR_BASE  # 224
EXTRAIT_FIRST_MAX = 0x8F        # premier octet 0x20-0x8F => Extrait, 2 octets
MAX_EXTRAITS = 4096


def address_length(first_byte: int) -> int:
    if first_byte < ADDR_BASE:
        raise ValueError(f"octet d'adresse invalide : {first_byte:#x}")
    return 2 if first_byte <= EXTRAIT_FIRST_MAX else 3


def encode_extrait_address(index: int) -> bytes:
    if not 0 <= index < MAX_EXTRAITS:
        raise ValueError(f"index d'Extrait hors plage : {index}")
    hi, lo = divmod(index, ADDR_RANGE)
    return bytes((ADDR_BASE + hi, ADDR_BASE + lo))


def decode_address(data: bytes) -> tuple[str, int]:
    if len(data) < 2:
        raise ValueError("adresse trop courte")
    first = data[0]
    if address_length(first) == 2:
        index = (first - ADDR_BASE) * ADDR_RANGE + (data[1] - ADDR_BASE)
        return ("extrait", index)
    if len(data) < 3:
        raise ValueError("adresse de Lexique : 3 octets attendus")
    index = ((first - 0x90) * ADDR_RANGE + (data[1] - ADDR_BASE)) * ADDR_RANGE + (data[2] - ADDR_BASE)
    return ("lexique", index)
