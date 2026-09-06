"""Mémoire de RELIS (spec §4.5) : état récurrent de taille constante + slots du Buffer."""
from dataclasses import dataclass, field
import torch


@dataclass
class State:
    gdn: list                      # par couche : (S, conv_state) ou None
    swa: list                      # par couche : dict cache ou None
    slots: torch.Tensor            # (B, K, d)
    pending: torch.Tensor          # (B, P, d) états du sommet en attente d'écriture
    seen: int = 0
    _init_slots: torch.Tensor = field(default=None, repr=False)

    @classmethod
    def init(cls, model, B: int, device) -> "State":
        gdn, swa = [], []
        for i, blk in enumerate(model.blocks):
            if blk.is_swa:
                gdn.append(None); swa.append(blk.mixer.init_cache(B, device))
            else:
                gdn.append(blk.mixer.init_state(B, device)); swa.append(None)
        slots = model.slots.unsqueeze(0).expand(B, -1, -1).contiguous().to(device)
        pending = torch.zeros(B, 0, model.cfg.d_model, device=device)
        return cls(gdn=gdn, swa=swa, slots=slots, pending=pending, seen=0, _init_slots=slots.detach().clone())

    def reset_memory(self) -> None:
        """REFRESH (spec §4.8) : Mémoire remise à zéro, Buffer conservé."""
        for i, g in enumerate(self.gdn):
            if g is not None:
                self.gdn[i] = (torch.zeros_like(g[0]), torch.zeros_like(g[1]))
        for i, c in enumerate(self.swa):
            if c is not None:
                self.swa[i] = {"k": c["k"][:, :, :0], "v": c["v"][:, :, :0], "pos": 0}
        self.pending = self.pending[:, :0]
        self.seen = 0

    def reset_all(self) -> None:
        """Nouveau tour utilisateur : tout est remis à zéro, slots compris."""
        self.reset_memory()
        self.slots = self._init_slots.clone()

    def size_bytes(self) -> int:
        n = 0
        for g in self.gdn:
            if g is not None:
                n += g[0].numel() * g[0].element_size() + g[1].numel() * g[1].element_size()
        for c in self.swa:
            if c is not None:
                n += c["k"].numel() * c["k"].element_size() + c["v"].numel() * c["v"].element_size()
        n += self.pending.numel() * self.pending.element_size()
        return n
