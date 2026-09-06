import torch
from relis.model.config import RelisConfig
from relis.model.relis import RelisModel
from relis.data.shards import ShardWriter
from relis.data.dataset import ByteWindows
from relis.train.loop import TrainConfig, lr_at, train, bits_per_byte


def _tiny_data(tmp_path):
    p = str(tmp_path / "train.bin")
    w = ShardWriter(p)
    w.add(b"le chat dort. le chien court. " * 400, "src=t")
    w.close()
    return ByteWindows(p, seq_len=32), ByteWindows(p, seq_len=32)


def _tcfg(**kw):
    base = dict(seq_len=32, batch_size=4, grad_accum=1, lr=3e-3, warmup_steps=5, max_steps=40,
                weight_decay=0.1, grad_clip=1.0, amp=False, ckpt_every_minutes=1e9, log_every=10)
    base.update(kw)
    return TrainConfig(**base)


def test_lr_schedule():
    c = _tcfg(lr=1.0, warmup_steps=10, max_steps=110)
    assert abs(lr_at(0, c)) < 1e-9
    assert abs(lr_at(10, c) - 1.0) < 1e-9
    assert 0.1 <= lr_at(110, c) <= 0.1 + 1e-9
    assert lr_at(60, c) < lr_at(20, c)


def test_loss_decreases_on_repetitive_data(tmp_path):
    torch.manual_seed(0)
    tr, va = _tiny_data(tmp_path)
    m = RelisModel(RelisConfig.tiny())
    losses = []
    out = train(m, _tcfg(), tr, va, str(tmp_path / "run"), device="cpu",
                resume=False, on_step=lambda s, l: losses.append(l))
    assert out["step"] == 40
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5 * 0.8
    assert out["val_bpb"] < 8.0


def test_resume_continues_from_checkpoint(tmp_path):
    tr, va = _tiny_data(tmp_path)
    m = RelisModel(RelisConfig.tiny())
    run = str(tmp_path / "run")
    train(m, _tcfg(max_steps=6), tr, va, run, device="cpu", resume=False)
    m2 = RelisModel(RelisConfig.tiny())
    seen = []
    out = train(m2, _tcfg(max_steps=9), tr, va, run, device="cpu", resume=True,
                on_step=lambda s, l: seen.append(s))
    assert seen == [7, 8, 9] and out["step"] == 9


def test_bits_per_byte_range(tmp_path):
    tr, _ = _tiny_data(tmp_path)
    m = RelisModel(RelisConfig.tiny())
    bpb = bits_per_byte(m, tr, seq_len=32, n_batches=2, batch_size=2, device="cpu")
    assert 6.0 < bpb < 10.0     # ~8 bits/octet pour un modèle non entraîné


def test_resume_calls_barrier_when_distributed(tmp_path, monkeypatch):
    import torch.distributed as dist

    calls = []
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "barrier", lambda *a, **kw: calls.append(1))

    tr, va = _tiny_data(tmp_path)
    m = RelisModel(RelisConfig.tiny())
    run = str(tmp_path / "run")
    train(m, _tcfg(max_steps=2), tr, va, run, device="cpu", resume=True)
    assert len(calls) == 1
