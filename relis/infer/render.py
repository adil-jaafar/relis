"""Rendu terminal du flux d'événements (spec §8) : le terminal est la démonstration.

Couleurs en ANSI direct, sans dépendance. Repli automatique quand la sortie n'est pas
un terminal, et option explicite pour forcer l'un ou l'autre.
"""
import codecs
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
        # Un caractère peut s'étaler sur jusqu'à quatre octets UTF-8 : le décodeur
        # incrémental retient les octets de tête en attente plutôt que de les
        # décoder (et donc les corrompre) un par un.
        self._dec = codecs.getincrementaldecoder("utf-8")("replace")
        self._inline = False       # vrai juste après un octet de génération streamé

    def _c(self, name: str, s: str) -> str:
        return f"{ANSI[name]}{s}{ANSI['reset']}" if self.color else s

    def _w(self, s: str = "") -> None:
        if not self.quiet:
            if self._inline:
                self.out.write("\n")
                self._inline = False
            self.out.write(s + "\n")
            self.out.flush()

    def handle(self, e) -> None:
        k = e.kind
        if k == "turn_start":
            self._w()
            self._w(self._c("bold", "? ") + e.text)
        elif k == "channel":
            n = e.size
            self._w(self._c("teal", f"  ⟨{e.channel}⟩ ") +
                    self._c("grey", f"{n} segment" + ("" if n == 1 else "s")))
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
                s = self._dec.decode(ch)
                if s:                   # rien à afficher tant qu'un caractère est incomplet
                    self.out.write(s)
                    self.out.flush()
                    self._inline = True
        elif k == "end":
            self._w()
        elif k == "turn_end":
            s = e.stats
            if self.quiet:
                self.out.write(e.text + "\n")
            else:
                self._w(self._c("grey",
                                f"  {s['read']} octets lus · {s['written']} écrits · "
                                f"{s['refresh']} relecture(s) · {round(s['saved'] * 100)} % "
                                f"du contexte économisé · {s['seconds']} s"))

    def run(self, events) -> str:
        for e in events:
            self.handle(e)
        return b"".join(self._answer).decode("utf-8", "replace")
