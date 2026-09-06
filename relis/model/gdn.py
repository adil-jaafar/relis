"""Gated DeltaNet en PyTorch pur (spec §4.4).

Convention vecteurs-lignes : S ∈ R^{dk×dv}, o_t = q_t S_t,
S_t = α_t S_{t-1} + β_t k_tᵀ (v_t − α_t k_t S_{t-1}).
Tout le cœur numérique est en fp32 ; l'état est toujours fp32.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RelisConfig
from .layers import RMSNorm, CausalConv1d


def gdn_step(q, k, v, log_alpha, beta, S):
    """Un pas de récurrence. q,k:(B,H,dk) v:(B,H,dv) log_alpha,beta:(B,H) S:(B,H,dk,dv)."""
    q, k, v, S = q.float(), k.float(), v.float(), S.float()
    a = log_alpha.float().exp()[..., None, None]
    b = beta.float()[..., None, None]
    kS = k.unsqueeze(-2) @ S                                  # (B,H,1,dv)
    S_new = a * S + b * (k.unsqueeze(-1) @ (v.unsqueeze(-2) - a * kS))
    o = (q.unsqueeze(-2) @ S_new).squeeze(-2)                 # (B,H,dv)
    return o, S_new


def gdn_chunked(q, k, v, log_alpha, beta, S, chunk: int):
    """Forme chunkée, équivalente à gdn_step appliqué L fois.

    q,k:(B,H,L,dk) (k L2-normalisé) v:(B,H,L,dv) log_alpha,beta:(B,H,L) S:(B,H,dk,dv)
    """
    q, k, v = q.float(), k.float(), v.float()
    log_alpha, beta, S = log_alpha.float(), beta.float(), S.float()
    B, H, L, dk = q.shape
    dv = v.shape[-1]
    C = chunk
    pad = (-L) % C
    if pad:
        # Positions de remplissage neutres : α=1 (log 0), β=0, k=q=v=0.
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        log_alpha = F.pad(log_alpha, (0, pad))
        beta = F.pad(beta, (0, pad))
    Lp = L + pad
    N = Lp // C
    q = q.view(B, H, N, C, dk)
    k = k.view(B, H, N, C, dk)
    v = v.view(B, H, N, C, dv)
    log_alpha = log_alpha.view(B, H, N, C)
    beta = beta.view(B, H, N, C)

    lg = torch.cumsum(log_alpha, dim=-1)                       # log γ_t dans le chunk
    gamma = lg.exp()                                           # (B,H,N,C)
    diff = lg.unsqueeze(-1) - lg.unsqueeze(-2)                 # (t,i) -> log(γ_t/γ_i)
    tri_strict = torch.ones(C, C, dtype=torch.bool, device=q.device).tril(-1)
    tri_incl = torch.ones(C, C, dtype=torch.bool, device=q.device).tril(0)
    ratio_strict = diff.masked_fill(~tri_strict, float("-inf")).exp()
    ratio_incl = diff.masked_fill(~tri_incl, float("-inf")).exp()

    KK = k @ k.transpose(-1, -2)                               # (B,H,N,C,C)
    M = ratio_strict * beta.unsqueeze(-2) * KK                 # β_i sur la colonne i
    eye = torch.eye(C, device=q.device, dtype=q.dtype)
    rhs = eye.expand_as(M).contiguous()
    T = torch.linalg.solve_triangular(eye + M, rhs, upper=False, unitriangular=True)
    U = T @ v                                                  # (B,H,N,C,dv)
    W = T @ (gamma.unsqueeze(-1) * k)                          # (B,H,N,C,dk)
    P = ratio_incl * beta.unsqueeze(-2) * (q @ k.transpose(-1, -2))
    gamma_C = gamma[..., -1]                                   # (B,H,N)
    decay_to_end = (lg[..., -1:] - lg).exp() * beta            # γ_C/γ_i · β_i

    outs = []
    for n in range(N):
        U_eff = U[:, :, n] - W[:, :, n] @ S                    # (B,H,C,dv)
        O = gamma[:, :, n, :, None] * (q[:, :, n] @ S) + P[:, :, n] @ U_eff
        S = gamma_C[:, :, n, None, None] * S + k[:, :, n].transpose(-1, -2) @ (decay_to_end[:, :, n, :, None] * U_eff)
        outs.append(O)
    o = torch.stack(outs, dim=2).reshape(B, H, Lp, dv)[:, :, :L]
    return o, S


class GatedDeltaNet(nn.Module):
    def __init__(self, cfg: RelisConfig):
        super().__init__()
        self.cfg = cfg
        d, H, dh = cfg.d_model, cfg.n_heads, cfg.head_dim
        inner = cfg.inner
        self.qkv = nn.Linear(d, 3 * inner, bias=False)
        self.conv = CausalConv1d(3 * inner, cfg.conv_kernel)
        self.beta_proj = nn.Linear(d, H, bias=True)
        self.dt_proj = nn.Linear(d, H, bias=True)
        # A_log initialisé comme dans Mamba-2 : log(1..H)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, H + 1, dtype=torch.float32)))
        with torch.no_grad():
            # dt_bias : softplus^{-1} d'un dt dans [1e-3, 1e-1]
            dt = torch.exp(torch.rand(H) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.gate_proj = nn.Linear(d, inner, bias=False)
        self.o_norm = RMSNorm(dh)
        self.o_proj = nn.Linear(inner, d, bias=False)

    def init_state(self, B: int, device):
        S = torch.zeros(B, self.cfg.n_heads, self.cfg.head_dim, self.cfg.head_dim, device=device, dtype=torch.float32)
        cs = torch.zeros(B, 3 * self.cfg.inner, self.cfg.conv_kernel - 1, device=device, dtype=torch.float32)
        return S, cs

    def _project(self, x, conv_state):
        B, L, _ = x.shape
        H, dh = self.cfg.n_heads, self.cfg.head_dim
        qkv, conv_state = self.conv(self.qkv(x), conv_state)
        qkv = F.silu(qkv)
        q, k, v = qkv.split(self.cfg.inner, dim=-1)
        q = q.view(B, L, H, dh).transpose(1, 2)
        k = k.view(B, L, H, dh).transpose(1, 2)
        v = v.view(B, L, H, dh).transpose(1, 2)
        q = F.normalize(q.float(), dim=-1)
        k = F.normalize(k.float(), dim=-1)
        beta = torch.sigmoid(self.beta_proj(x).float()).transpose(1, 2)               # (B,H,L)
        log_alpha = -self.A_log.float().exp()[None, :, None] * F.softplus(self.dt_proj(x).float()).transpose(1, 2)
        return q, k, v.float(), log_alpha, beta, conv_state

    def _output(self, o, x):
        B, H, L, dh = o.shape
        o = self.o_norm(o).transpose(1, 2).reshape(B, L, H * dh)
        gate = F.silu(self.gate_proj(x).float())
        return self.o_proj((o * gate).to(x.dtype))

    def forward(self, x, S, conv_state):
        if S is None:
            S, conv_state = self.init_state(x.shape[0], x.device)
        q, k, v, log_alpha, beta, conv_state = self._project(x, conv_state)
        o, S = gdn_chunked(q, k, v, log_alpha, beta, S, self.cfg.chunk)
        return self._output(o, x), S, conv_state

    def step(self, x, S, conv_state):
        q, k, v, log_alpha, beta, conv_state = self._project(x, conv_state)
        o, S = gdn_step(q[:, :, 0], k[:, :, 0], v[:, :, 0], log_alpha[:, :, 0], beta[:, :, 0], S)
        return self._output(o.unsqueeze(2), x), S, conv_state
