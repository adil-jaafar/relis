import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMSNorm calculée en fp32 (spec §6.4), sortie au dtype d'entrée."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d: int, mult: int):
        super().__init__()
        h = mult * d
        self.w1 = nn.Linear(d, h, bias=False)
        self.w3 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(h, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CausalConv1d(nn.Module):
    """Convolution depthwise causale courte, avec état pour le mode pas-à-pas.

    x : (B, L, C). conv_state : (B, C, kernel-1) = dernières entrées vues.
    """

    def __init__(self, channels: int, kernel: int):
        super().__init__()
        self.channels = channels
        self.kernel = kernel
        self.conv = nn.Conv1d(channels, channels, kernel, groups=channels, bias=True)

    def forward(self, x: torch.Tensor, conv_state: torch.Tensor | None):
        B, L, C = x.shape
        xt = x.transpose(1, 2)  # (B, C, L)
        if conv_state is None:
            conv_state = xt.new_zeros(B, C, self.kernel - 1)
        xcat = torch.cat([conv_state.to(xt.dtype), xt], dim=2)  # (B, C, k-1+L)
        y = self.conv(xcat)  # (B, C, L)
        new_state = xcat[:, :, -(self.kernel - 1):].float()   # l'état reste fp32 (spec §6.4)
        return y.transpose(1, 2), new_state
