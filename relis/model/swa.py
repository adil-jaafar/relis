"""Attention à fenêtre glissante (spec §4.4) : opérateur local, cache borné à window-1.

Biais ALiBi (relatif) : la position absolue ne sert qu'à calculer i-j.
"""
import torch
import torch.nn as nn

from .config import RelisConfig


def windowed_attention(q, k_all, v_all, qi, kj, slopes, window, dh, seg_mask=None):
    """Cœur d'attention local, fp32 garanti sous autocast.

    seg_mask : bool (B,1,L,Lk), True si requête et clé sont dans le même segment
    (aucun REFRESH entre elles). None = un seul segment.
    """
    with torch.autocast(device_type=q.device.type, enabled=False):
        allowed = (kj <= qi) & (kj > qi - window)                    # (L, Lk)
        if seg_mask is not None:
            allowed = allowed[None, None] & seg_mask                   # (B,1,L,Lk)
        scores = (q @ k_all.transpose(-1, -2)) / dh ** 0.5           # (B,H,L,Lk)
        scores = scores - slopes[None, :, None, None] * (qi - kj).float()
        scores = scores.masked_fill(~allowed, float("-inf"))
        o = torch.softmax(scores, dim=-1) @ v_all
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
        return {"k": torch.zeros(B, H, 0, dh, device=device), "v": torch.zeros(B, H, 0, dh, device=device),
                "pos": 0, "seg": torch.zeros(B, dtype=torch.long, device=device),
                "kseg": torch.zeros(B, 0, dtype=torch.long, device=device)}

    def forward(self, x, cache, reset=None):
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
        qi = torch.arange(pos0, pos0 + L, device=x.device)[:, None]
        kj = torch.arange(pos0 - Lc, pos0 + L, device=x.device)[None, :]
        # segments : un REFRESH à la position t ouvre un nouveau segment à partir de t
        inc = torch.zeros(B, L, dtype=torch.long, device=x.device) if reset is None \
            else reset.to(torch.long).reshape(B, L).cumsum(dim=1)
        qseg = cache["seg"][:, None] + inc                             # (B,L)
        kseg_all = torch.cat([cache["kseg"], qseg], dim=1)             # (B,Lc+L)
        seg_mask = kseg_all[:, None, None, :] == qseg[:, None, :, None]  # (B,1,L,Lc+L)
        o = windowed_attention(q, k_all, v_all, qi, kj, self.slopes, W, dh, seg_mask)
        y = self.o_proj(o.transpose(1, 2).reshape(B, L, H * dh).to(x.dtype))
        keep = W - 1
        new_cache = {"k": k_all[:, :, -keep:] if keep > 0 else k_all[:, :, :0],
                     "v": v_all[:, :, -keep:] if keep > 0 else v_all[:, :, :0],
                     "pos": pos0 + L, "seg": qseg[:, -1],
                     "kseg": kseg_all[:, -keep:] if keep > 0 else kseg_all[:, :0]}
        return y, new_cache

    def step(self, x, cache, reset=None):
        return self.forward(x, cache, reset)
