"""Assemblage de RELIS (spec §4.1, §4.4, §4.6) : un réseau, trois modes.

forward() traite la séquence par blocs de cfg.block positions ; dans un bloc,
toutes les positions lisent les slots tels qu'ils étaient au début du bloc ;
à la frontière, SlotWrite met à jour les slots chunk par chunk. step() suit
exactement le même calendrier, d'où l'équivalence numérique des deux modes.
"""
import torch
import torch.nn as nn

from .config import RelisConfig
from .layers import RMSNorm, SwiGLU
from .gdn import GatedDeltaNet
from .swa import SlidingWindowAttention
from .buffer import SlotRead, SlotWrite, init_slots
from .state import State


class Block(nn.Module):
    def __init__(self, cfg: RelisConfig, is_swa: bool):
        super().__init__()
        self.is_swa = is_swa
        self.norm1 = RMSNorm(cfg.d_model)
        self.mixer = SlidingWindowAttention(cfg) if is_swa else GatedDeltaNet(cfg)
        self.read = SlotRead(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_mult)

    def forward(self, x, st_gdn, st_swa, slots, single_step: bool = False):
        h = self.norm1(x)
        if self.is_swa:
            y, st_swa = (self.mixer.step if single_step else self.mixer.forward)(h, st_swa)
        else:
            S, cs = st_gdn
            y, S, cs = (self.mixer.step if single_step else self.mixer.forward)(h, S, cs)
            st_gdn = (S, cs)
        x = x + y
        x = x + self.read(x, slots)
        x = x + self.mlp(self.norm2(x))
        return x, st_gdn, st_swa


class RelisModel(nn.Module):
    def __init__(self, cfg: RelisConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.d_model)
        self.mode_embed = nn.Embedding(cfg.n_modes, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, cfg.is_swa(i)) for i in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.slots = init_slots(cfg)
        self.writer = SlotWrite(cfg)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.mode_embed.weight, std=0.02)

    def head_weight(self) -> torch.Tensor:
        return self.embed.weight   # tête liée (spec §4.4)

    def new_state(self, B: int, device) -> State:
        return State.init(self, B, device)

    def _run_piece(self, x_ids, mode_ids, state: State, single_step: bool):
        """Passe un morceau (≤ block positions, sans frontière interne) dans la pile."""
        h = self.embed(x_ids) + self.mode_embed(mode_ids)
        for i, blk in enumerate(self.blocks):
            h, g, s = blk(h, state.gdn[i], state.swa[i], state.slots, single_step=single_step)
            state.gdn[i], state.swa[i] = g, s
        top = self.final_norm(h)
        logits = top @ self.head_weight().t().to(top.dtype)
        # écriture différée : accumuler, écrire à la frontière de bloc
        state.pending = torch.cat([state.pending, top.float()], dim=1)
        state.seen += x_ids.shape[1]
        if state.pending.shape[1] == self.cfg.block:
            state.slots = self.writer.apply_blocks(state.pending.to(state.slots.dtype), state.slots)
            state.pending = state.pending[:, :0]
        return logits, state

    def forward(self, x, mode, state: State):
        B, L = x.shape
        if isinstance(mode, int):
            mode = torch.full((B, L), mode, dtype=torch.long, device=x.device)
        logits = []
        t = 0
        while t < L:
            room = self.cfg.block - state.pending.shape[1]
            n = min(room, L - t)
            lg, state = self._run_piece(x[:, t:t + n], mode[:, t:t + n], state, single_step=False)
            logits.append(lg)
            t += n
        return torch.cat(logits, dim=1), state

    def step(self, x, mode, state: State):
        B = x.shape[0]
        if isinstance(mode, int):
            mode = torch.full((B,), mode, dtype=torch.long, device=x.device)
        lg, state = self._run_piece(x[:, None], mode[:, None], state, single_step=True)
        return lg[:, 0], state
