"""Attention à fenêtre glissante (spec §4.4) : opérateur local, cache borné à window-1.

Biais ALiBi (relatif) : la position absolue ne sert qu'à calculer i-j.
"""
import torch
import torch.nn as nn

from .config import RelisConfig


def windowed_attention(q, k_all, v_all, qi, kj, slopes, window, dh):
    """Cœur d'attention local : calcul en fp32 garanti sous autocast.

    Args:
        q: (B, H, L, dh) - queries, float32
        k_all: (B, H, Lc+L, dh) - concatenated cached + current keys, float32
        v_all: (B, H, Lc+L, dh) - concatenated cached + current values, float32
        qi: (L, 1) - positions absolues des requêtes
        kj: (1, Lc+L) - positions absolues des clés
        slopes: (H,) - ALiBi slopes par tête
        window: int - taille de fenêtre
        dh: int - dimension par tête

    Returns:
        o: (B, H, L, dh) - attention output, float32
    """
    with torch.autocast(device_type=q.device.type, enabled=False):
        allowed = (kj <= qi) & (kj > qi - window)
        scores = (q @ k_all.transpose(-1, -2)) / dh ** 0.5           # (B,H,L,Lc+L)
        scores = scores - slopes[None, :, None, None] * (qi - kj).float()
        scores = scores.masked_fill(~allowed, float("-inf"))
        o = torch.softmax(scores, dim=-1) @ v_all                    # fp32
    return o


class SlidingWindowAttention(nn.Module):
    def __init__(self, cfg: RelisConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.inner, bias=False)
        self.o_proj = nn.Linear(cfg.inner, cfg.d_model, bias=False)
        H = cfg.n_heads
        slopes = torch.tensor([2.0 ** (-8.0 * (h + 1) / H) for h in range(H)], dtype=torch.float32)
        self.register_buffer("slopes", slopes, persistent=False)

    def init_cache(self, B: int, device):
        H, dh = self.cfg.n_heads, self.cfg.head_dim
        return {"k": torch.zeros(B, H, 0, dh, device=device), "v": torch.zeros(B, H, 0, dh, device=device), "pos": 0}

    def forward(self, x, cache):
        B, L, _ = x.shape
        H, dh, W = self.cfg.n_heads, self.cfg.head_dim, self.cfg.window
        if cache is None:
            cache = self.init_cache(B, x.device)
        q, k, v = self.qkv(x).split(self.cfg.inner, dim=-1)
        q = q.view(B, L, H, dh).transpose(1, 2).float()
        k = k.view(B, L, H, dh).transpose(1, 2).float()
        v = v.view(B, L, H, dh).transpose(1, 2).float()
        k_all = torch.cat([cache["k"].float(), k], dim=2)
        v_all = torch.cat([cache["v"].float(), v], dim=2)
        Lc = cache["k"].shape[2]
        pos0 = cache["pos"]
        # positions absolues des requêtes et des clés
        qi = torch.arange(pos0, pos0 + L, device=x.device)[:, None]
        kj = torch.arange(pos0 - Lc, pos0 + L, device=x.device)[None, :]
        o = windowed_attention(q, k_all, v_all, qi, kj, self.slopes, W, dh)
        y = self.o_proj(o.transpose(1, 2).reshape(B, L, H * dh).to(x.dtype))
        keep = W - 1
        new_cache = {"k": k_all[:, :, -keep:] if keep > 0 else k_all[:, :, :0],
                     "v": v_all[:, :, -keep:] if keep > 0 else v_all[:, :, :0],
                     "pos": pos0 + L}
        return y, new_cache

    def step(self, x, cache):
        return self.forward(x, cache)
