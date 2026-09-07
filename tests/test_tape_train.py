import os
import torch
from relis.model.config import RelisConfig
from relis.model.relis import RelisModel
from relis.tape import codes as C
from relis.data.pack import build_shards, TapeWindows
from relis.data.shards import ShardWriter
from relis.train.loop import TrainConfig, train, _loss, load_weights
from relis.train.metrics import decision_accuracy
from relis.train.checkpoint import save_checkpoint


def _tapes(tmp_path, seq_len=2048, block=32, episodes=60):
    pre = str(tmp_path / "pre.bin")
    w = ShardWriter(pre); w.add(("le chat dort. " * 400).encode(), "src=t"); w.close()
    out = str(tmp_path / "tapes")
    build_shards(out, episodes=episodes, seed=0, seq_len=seq_len, block=block, replay=pre, replay_frac=0.2, val_frac=0.1)
    return TapeWindows(os.path.join(out, "train")), TapeWindows(os.path.join(out, "val"))


def _tcfg(**kw):
    base = dict(seq_len=2047, batch_size=2, grad_accum=1, lr=2e-3, warmup_steps=3, max_steps=15,
                weight_decay=0.1, grad_clip=1.0, amp=False, ckpt_every_minutes=1e9, log_every=5)
    base.update(kw)
    return TrainConfig(**base)


def test_weighted_loss_ignores_zero_weight_positions(tmp_path):
    tr, _ = _tapes(tmp_path)
    m = RelisModel(RelisConfig.tiny())
    b = tr.sample(2, torch.Generator().manual_seed(0))
    l1 = _loss(m, b, "cpu", amp=False)
    b2 = {k: v.clone() for k, v in b.items()}
    zero = b2["w"] == 0
    b2["y"][zero] = (b2["y"][zero] + 7) % 256           # change les cibles là où le poids est nul
    l2 = _loss(m, b2, "cpu", amp=False)
    assert zero.any() and torch.allclose(l1, l2, atol=1e-6)


def test_loss_accepts_plain_tuple_batches():
    m = RelisModel(RelisConfig.tiny())
    x = torch.randint(0, 256, (1, 40)); y = torch.randint(0, 256, (1, 40))
    l = _loss(m, (x, y), "cpu", amp=False)
    assert torch.isfinite(l)


class _Oracle:
    """Faux modèle : prédit exactement la cible (logits one-hot)."""
    def __init__(self): self.cfg = RelisConfig.tiny(); self._y = None
    def eval(self): return self
    def train(self): return self
    def new_state(self, B, device): return None
    def __call__(self, x, mode, state, reset=None, slots_reset=None):
        return torch.nn.functional.one_hot(self._y, 256).float() * 10, state


def test_decision_accuracy_on_oracle_and_random(tmp_path):
    tr, _ = _tapes(tmp_path)
    o = _Oracle()
    import relis.train.metrics as M
    orig = tr.sample
    def spy(bs, g):
        b = orig(bs, g); o._y = b["y"]; return b
    tr.sample = spy
    r = M.decision_accuracy(o, tr, n_batches=2, batch_size=2, device="cpu")
    assert r["n"] > 0 and abs(r["acc"] - 1.0) < 1e-9
    assert set(r["per_code"]) <= {C.name(c) for c in C.DECISION_CODES}
    tr.sample = orig
    r2 = M.decision_accuracy(RelisModel(RelisConfig.tiny()), tr, n_batches=2, batch_size=2, device="cpu")
    assert 0.0 <= r2["acc"] <= 1.0 and r2["n"] > 0


def test_tape_training_decreases_loss_and_reports_decisions(tmp_path):
    torch.manual_seed(0)
    tr, va = _tapes(tmp_path)
    m = RelisModel(RelisConfig.tiny())
    losses, seen = [], []
    out = train(m, _tcfg(), tr, va, str(tmp_path / "run"), device="cpu", resume=False,
                on_step=lambda s, l: losses.append(l),
                extra_val=lambda mm: seen.append(decision_accuracy(mm, va, 1, 2, "cpu")) or seen[-1])
    assert out["step"] == 15 and losses[-1] < losses[0]
    assert seen and "acc" in seen[-1]


def test_init_from_loads_weights_only(tmp_path):
    cfg = RelisConfig.tiny()
    src = RelisModel(cfg)
    save_checkpoint(str(tmp_path / "src"), src, None, None, step=42, cfg_dict=cfg.to_dict(), extra={})
    dst = RelisModel(cfg)
    step = load_weights(dst, str(tmp_path / "src" / "last.pt"))
    assert step == 42
    for a, b in zip(src.parameters(), dst.parameters()):
        assert torch.equal(a, b)
    tr, va = _tapes(tmp_path)
    out = train(dst, _tcfg(max_steps=2, init_from=str(tmp_path / "src" / "last.pt")), tr, va,
                str(tmp_path / "run2"), device="cpu", resume=True)
    assert out["step"] == 2      # init_from n'impose pas le pas source : le run repart de 0
