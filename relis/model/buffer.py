"""Buffer d'intention (spec §4.6) : K slots partagés par toutes les couches.

Lecture : chaque bloc fait une cross-attention positions -> slots.
Écriture : au sommet de la pile, une fois par chunk de 64 octets, slots <- slots + g ⊙ Δ.
"""
import torch
import torch.nn as nn

from .config import RelisConfig
from .layers import RMSNorm


def init_slots(cfg: RelisConfig) -> nn.Parameter:
    return nn.Parameter(torch.randn(cfg.n_slots, cfg.d_model) * 0.02)


def expand_slots(param: torch.Tensor, B: int) -> torch.Tensor:
    return param.unsqueeze(0).expand(B, -1, -1).contiguous()


def _mha(q, k, v, n_heads):
    """q:(B,Lq,d) k,v:(B,Lk,d) -> (B,Lq,d), softmax en fp32."""
    B, Lq, d = q.shape
    Lk = k.shape[1]
    dh = d // n_heads
    q = q.view(B, Lq, n_heads, dh).transpose(1, 2).float()
    k = k.view(B, Lk, n_heads, dh).transpose(1, 2).float()
    v = v.view(B, Lk, n_heads, dh).transpose(1, 2).float()

    with torch.autocast(device_type=q.device.type, enabled=False):
        scores = (q @ k.transpose(-1, -2)) / dh ** 0.5
        att = torch.softmax(scores, dim=-1)
        o = att @ v

    return o.transpose(1, 2).reshape(B, Lq, d)


class SlotRead(nn.Module):
    def __init__(self, cfg: RelisConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        r = cfg.inner // 2            # dimension interne réduite (256 en V1)
        assert r % cfg.n_heads == 0
        self.norm_x = RMSNorm(d)
        self.norm_s = RMSNorm(d)
        self.q = nn.Linear(d, r, bias=False)
        self.kv = nn.Linear(d, 2 * r, bias=False)
        self.o = nn.Linear(r, d, bias=False)

    def forward(self, x, slots):
        k, v = self.kv(self.norm_s(slots)).chunk(2, dim=-1)
        y = _mha(self.q(self.norm_x(x)), k, v, self.cfg.n_heads)
        return self.o(y.to(x.dtype))


class SlotWrite(nn.Module):
    def __init__(self, cfg: RelisConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.norm_s = RMSNorm(d)
        self.norm_h = RMSNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.kv = nn.Linear(d, 2 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(2 * d, d, bias=True)
        with torch.no_grad():
            self.gate.bias.fill_(-2.0)   # écritures prudentes au départ

    def forward(self, h_chunk, slots):
        k, v = self.kv(self.norm_h(h_chunk)).chunk(2, dim=-1)
        delta = self.o(_mha(self.q(self.norm_s(slots)), k, v, self.cfg.n_heads).to(slots.dtype))
        g = torch.sigmoid(self.gate(torch.cat([slots, delta], dim=-1)).float())
        return (slots.float() + g * delta.float()).to(slots.dtype)

    def apply_blocks(self, h, slots):
        L = h.shape[1]
        C = self.cfg.chunk
        assert L % C == 0, "apply_blocks attend un multiple de chunk"
        for c in range(L // C):
            slots = self.forward(h[:, c * C:(c + 1) * C], slots)
        return slots
