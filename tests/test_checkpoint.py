import torch
from relis.model.config import RelisConfig
from relis.model.relis import RelisModel
from relis.train.checkpoint import save_checkpoint, load_checkpoint


def test_roundtrip(tmp_path):
    cfg = RelisConfig.tiny()
    m = RelisModel(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randint(0, 256, (1, 8))
    logits, _ = m(x, 1, m.new_state(1, "cpu"))
    logits.sum().backward(); opt.step()
    path = save_checkpoint(str(tmp_path), m, opt, None, step=7, cfg_dict=cfg.to_dict(), extra={"note": "ok"})
    assert path.endswith("last.pt")

    m2 = RelisModel(cfg)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    info = load_checkpoint(str(tmp_path), m2, opt2)
    assert info["step"] == 7 and info["extra"]["note"] == "ok" and info["cfg"]["d_model"] == 64
    for a, b in zip(m.parameters(), m2.parameters()):
        assert torch.equal(a, b)
    assert opt2.state_dict()["state"].keys() == opt.state_dict()["state"].keys()


def test_load_missing_returns_none(tmp_path):
    m = RelisModel(RelisConfig.tiny())
    assert load_checkpoint(str(tmp_path), m) is None


def test_load_skips_cuda_rng_on_device_mismatch(tmp_path):
    import os
    cfg = RelisConfig.tiny()
    m = RelisModel(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randint(0, 256, (1, 8))
    logits, _ = m(x, 1, m.new_state(1, "cpu"))
    logits.sum().backward(); opt.step()
    path = save_checkpoint(str(tmp_path), m, opt, None, step=42, cfg_dict=cfg.to_dict(), extra={"note": "test"})

    # Simulate checkpoint saved on 2 GPUs by modifying the saved file
    checkpoint_path = os.path.join(str(tmp_path), "last.pt")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload["rng"]["cuda"] = [torch.ByteTensor(256), torch.ByteTensor(256)]  # Two fake GPU states
    torch.save(payload, checkpoint_path)

    # Load on CPU-only machine should not crash and should skip CUDA RNG
    m2 = RelisModel(cfg)
    info = load_checkpoint(str(tmp_path), m2)
    assert info["step"] == 42
    assert info["extra"]["note"] == "test"
