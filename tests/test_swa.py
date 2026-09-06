import torch
from relis.model.config import RelisConfig
from relis.model.swa import SlidingWindowAttention, windowed_attention


def _naive(m, x):
    """Référence : attention complète masquée par fenêtre, ALiBi, en une passe."""
    B, L, _ = x.shape
    H, dh, W = m.cfg.n_heads, m.cfg.head_dim, m.cfg.window
    q, k, v = m.qkv(x).split(m.cfg.inner, dim=-1)
    q = q.view(B, L, H, dh).transpose(1, 2).float()
    k = k.view(B, L, H, dh).transpose(1, 2).float()
    v = v.view(B, L, H, dh).transpose(1, 2).float()
    i = torch.arange(L)[:, None]; j = torch.arange(L)[None, :]
    allowed = (j <= i) & (j > i - W)
    scores = (q @ k.transpose(-1, -2)) / dh ** 0.5
    scores = scores - m.slopes[None, :, None, None] * (i - j).float()
    scores = scores.masked_fill(~allowed, float("-inf"))
    o = torch.softmax(scores, dim=-1) @ v
    return m.o_proj(o.transpose(1, 2).reshape(B, L, H * dh))


def test_forward_matches_naive():
    cfg = RelisConfig.tiny()          # window 16
    m = SlidingWindowAttention(cfg)
    x = torch.randn(2, 45, cfg.d_model)
    y, cache = m(x, None)
    assert torch.allclose(y, _naive(m, x), atol=1e-5)
    assert cache["k"].shape[2] == cfg.window - 1
    assert cache["pos"] == 45


def test_forward_split_equals_whole():
    cfg = RelisConfig.tiny()
    m = SlidingWindowAttention(cfg)
    x = torch.randn(1, 50, cfg.d_model)
    y_all, c_all = m(x, None)
    y1, c = m(x[:, :7], None)
    y2, c = m(x[:, 7:30], c)
    y3, c = m(x[:, 30:], c)
    assert torch.allclose(torch.cat([y1, y2, y3], dim=1), y_all, atol=1e-5)
    assert torch.allclose(c["k"], c_all["k"], atol=1e-6)


def test_steps_equal_forward():
    cfg = RelisConfig.tiny()
    m = SlidingWindowAttention(cfg)
    x = torch.randn(2, 33, cfg.d_model)
    y_all, _ = m(x, None)
    c = m.init_cache(2, x.device)
    ys = []
    for t in range(33):
        y_t, c = m.step(x[:, t:t + 1], c)
        ys.append(y_t)
    assert torch.allclose(torch.cat(ys, dim=1), y_all, atol=1e-5)


def test_cache_is_bounded():
    cfg = RelisConfig.tiny()
    m = SlidingWindowAttention(cfg)
    _, c = m(torch.randn(1, 500, cfg.d_model), None)
    assert c["k"].shape[2] == cfg.window - 1 and c["v"].shape[2] == cfg.window - 1


def test_attention_core_fp32_under_autocast():
    """Vérifie que le cœur d'attention reste fp32 même sous autocast bf16."""
    cfg = RelisConfig.tiny()
    device = torch.device("cpu")

    # Inputs fp32
    B, L, Lc, H, dh = 1, 20, 5, cfg.n_heads, cfg.head_dim
    q = torch.randn(B, H, L, dh, dtype=torch.float32, device=device)
    k_all = torch.randn(B, H, Lc + L, dh, dtype=torch.float32, device=device)
    v_all = torch.randn(B, H, Lc + L, dh, dtype=torch.float32, device=device)
    slopes = torch.tensor([2.0 ** (-8.0 * (h + 1) / H) for h in range(H)], dtype=torch.float32, device=device)
    qi = torch.arange(Lc, Lc + L, device=device)[:, None]
    kj = torch.arange(0, Lc + L, device=device)[None, :]

    # Test core function inside and outside autocast
    o_plain = windowed_attention(q, k_all, v_all, qi, kj, slopes, cfg.window, dh)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        o_ac = windowed_attention(q, k_all, v_all, qi, kj, slopes, cfg.window, dh)

    # Core should always return fp32
    assert o_plain.dtype == torch.float32, f"Expected float32, got {o_plain.dtype}"
    assert o_ac.dtype == torch.float32, f"Expected float32 under autocast, got {o_ac.dtype}"
    assert torch.allclose(o_ac, o_plain, atol=1e-5), "Core attention results differ under autocast"

    # Test module forward under autocast
    m = SlidingWindowAttention(cfg)
    x = torch.randn(1, 20, cfg.d_model, dtype=torch.float32, device=device)

    y_plain, c_plain = m(x, None)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        y_ac, c_ac = m(x, None)

    # Cache k,v should be fp32
    assert c_plain["k"].dtype == torch.float32, f"Expected cache k float32, got {c_plain['k'].dtype}"
    assert c_ac["k"].dtype == torch.float32, f"Expected cache k float32 under autocast, got {c_ac['k'].dtype}"

    # Output can be in any dtype from projections, but should be close
    # Allow larger tolerance since o_proj may run in bf16
    assert torch.allclose(y_ac.float(), y_plain.float(), atol=5e-2), "Module output differs under autocast"
