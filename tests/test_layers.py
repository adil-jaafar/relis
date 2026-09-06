import torch
from relis.model.config import RelisConfig
from relis.model.layers import RMSNorm, SwiGLU, CausalConv1d


def test_config_defaults_match_spec():
    c = RelisConfig()
    assert (c.d_model, c.n_layers, c.n_heads, c.head_dim) == (768, 16, 8, 64)
    assert (c.window, c.chunk, c.block, c.n_slots) == (512, 64, 512, 32)
    assert [i for i in range(c.n_layers) if c.is_swa(i)] == [3, 7, 11, 15]
    assert c.inner == 512


def test_tiny_config_is_small_and_consistent():
    c = RelisConfig.tiny()
    assert c.d_model == 64 and c.n_layers == 4
    assert c.block % c.chunk == 0
    assert sum(c.is_swa(i) for i in range(c.n_layers)) == 1


def test_rmsnorm_fp32_and_shape():
    n = RMSNorm(8)
    x = torch.randn(2, 5, 8, dtype=torch.float16)
    y = n(x)
    assert y.dtype == torch.float16 and y.shape == x.shape
    ref = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    assert torch.allclose(y.float(), ref, atol=1e-2)


def test_swiglu_shape():
    m = SwiGLU(8, 2)
    assert m(torch.randn(3, 8)).shape == (3, 8)


def test_causal_conv_matches_streaming():
    conv = CausalConv1d(6, 4)
    x = torch.randn(2, 11, 6)
    y_full, st_full = conv(x, None)
    ys, st = [], None
    for t in range(11):
        y_t, st = conv(x[:, t:t + 1], st)
        ys.append(y_t)
    y_stream = torch.cat(ys, dim=1)
    assert torch.allclose(y_full, y_stream, atol=1e-6)
    assert torch.allclose(st_full, st, atol=1e-6)
    assert st.shape == (2, 6, 3)


def test_causal_conv_is_causal():
    conv = CausalConv1d(4, 4)
    x = torch.randn(1, 9, 4)
    y1, _ = conv(x, None)
    x2 = x.clone(); x2[:, 6] += 10.0
    y2, _ = conv(x2, None)
    assert torch.allclose(y1[:, :6], y2[:, :6])
    assert not torch.allclose(y1[:, 6], y2[:, 6])
