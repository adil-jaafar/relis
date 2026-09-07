"""En-têtes de segment (spec §4.3) : `clé=valeur;clé=valeur`, valeurs échappées."""
from . import codes as C

MAX_BYTES = 120


def _clean(value) -> str:
    s = str(value)
    for ch in (";", "=", "\t", "\n", "\r"):   # séparateurs et sauts de ligne
        s = s.replace(ch, " ")
    return C.sanitize(s.encode("utf-8")).decode("utf-8", errors="replace")


def format_header(fields: dict) -> str:
    s = ";".join(f"{k}={_clean(v)}" for k, v in fields.items())
    return s.encode("utf-8")[:MAX_BYTES].decode("utf-8", errors="ignore")


def parse_header(s: str) -> dict:
    out = {}
    for part in s.split(";"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        if k:
            out[k] = v
    return out
