import json, os
import numpy as np
import torch
from relis.data.episodes import iter_episodes
from relis.data.pack import (pad_tape, pack_tapes, replay_tape, TapeShardWriter, TapeWindows,
                             PAD, FLAG_RESET, FLAG_SLOTS, build_shards, implied_epochs)
from relis.data.shards import ShardWriter
from relis.tape.tape import build_tape, WEIGHTS, W_LOW, W_ONE


def test_pad_tape_aligns_to_block():
    t = build_tape(next(iter_episodes(0, 1)))
    p = pad_tape(t, 32)
    assert len(p) % 32 == 0 and len(p) - len(t) < 32
    assert all(b == PAD for b in p.data[len(t):]) and all(w == 0 for w in p.wclass[len(t):])


def test_pack_sets_flags_at_tape_starts_on_block_multiples():
    tapes = [build_tape(s) for s in iter_episodes(1, 40)]
    stats = {}
    seqs = list(pack_tapes(tapes, seq_len=4096, block=32, stats=stats))
    assert seqs and stats["skipped"] >= 0
    for s in seqs:
        assert s["data"].shape == (4096,) and s["data"].dtype == np.uint8
        starts = np.flatnonzero(s["flags"] & FLAG_SLOTS)
        assert len(starts) == s["n_tapes"] and starts[0] == 0
        assert all(st % 32 == 0 for st in starts)
        has_reset = (s["flags"] & FLAG_RESET) != 0
        has_slots = (s["flags"] & FLAG_SLOTS) != 0
        assert not (has_slots & ~has_reset).any()          # un début de ruban porte toujours le bit reset
        internal = np.flatnonzero(has_reset & ~has_slots)   # REFRESH internes : reset sans slots
        assert all(st not in set(starts.tolist()) for st in internal)


def test_replay_tape_is_full_length_weight_one(tmp_path):
    p = str(tmp_path / "train.bin")
    w = ShardWriter(p); w.add(bytes(range(256)) * 40, "src=t"); w.close()
    import random
    t = replay_tape(p, 512, random.Random(0), W_ONE)
    assert len(t) == 512 and set(t.wclass) == {W_ONE} and not any(t.reset)


def test_replay_default_weight_is_low(tmp_path):
    p = str(tmp_path / "train.bin")
    w = ShardWriter(p); w.add(bytes(range(256)) * 40, "src=t"); w.close()
    import random
    t = replay_tape(p, 512, random.Random(0))
    assert len(t) == 512 and set(t.wclass) == {W_LOW}
    assert all(abs(x - WEIGHTS[W_LOW]) < 1e-9 for x in t.weights())


def test_replay_short_file_raises(tmp_path):
    p = str(tmp_path / "court.bin")
    w = ShardWriter(p); w.add(b"a" * 100, "src=t"); w.close()
    import random, pytest
    with pytest.raises(ValueError, match="trop court"):
        replay_tape(p, 512, random.Random(0))


def test_writer_and_dataset_roundtrip(tmp_path):
    tapes = [build_tape(s) for s in iter_episodes(2, 30)]
    out = str(tmp_path / "tapes")
    wr = TapeShardWriter(out)
    n = 0
    for s in pack_tapes(tapes, seq_len=2048, block=32):
        wr.add(s); n += 1
    wr.close()
    meta = json.load(open(os.path.join(out, "meta.json")))
    assert meta["n"] == n and meta["seq_len"] == 2048
    ds = TapeWindows(out)
    assert len(ds) == n and ds.has_decisions
    b = ds.sample(3, torch.Generator().manual_seed(0))
    assert b["x"].shape == (3, 2047) and b["y"].shape == (3, 2047)
    assert torch.equal(b["x"][:, 1:], b["y"][:, :-1])
    assert b["w"].dtype == torch.float32
    uniq = b["w"].unique().tolist()
    assert all(any(abs(u - w) < 1e-6 for w in WEIGHTS) for u in uniq)
    assert b["reset"].dtype == torch.bool and b["slots_reset"].dtype == torch.bool
    pos = torch.nonzero(b["slots_reset"])[:, 1]
    assert (pos % 32 == 0).all()


def test_build_shards_cli_function(tmp_path):
    p = str(tmp_path / "pre.bin")
    w = ShardWriter(p); w.add(("le chat dort. " * 300).encode(), "src=t"); w.close()
    out = str(tmp_path / "tapes")
    stats = build_shards(out, episodes=60, seed=0, seq_len=2048, block=32, replay=p, replay_frac=0.2, val_frac=0.1)
    assert stats["train_seqs"] > 0 and stats["val_seqs"] > 0
    assert os.path.exists(os.path.join(out, "train", "data.bin")) and os.path.exists(os.path.join(out, "val", "meta.json"))


def test_build_shards_reports_mix(tmp_path):
    p = str(tmp_path / "pre.bin")
    w = ShardWriter(p); w.add(("le chat dort. " * 600).encode(), "src=t"); w.close()
    out = str(tmp_path / "tapes")
    stats = build_shards(out, episodes=120, seed=0, seq_len=2048, block=32,
                         replay=p, replay_frac=0.2, val_frac=0.1)
    for k in ("tape_seqs", "replay_seqs", "sum_w_decisions", "sum_w_answer", "sum_w_low",
              "sum_w_replay", "share_decisions", "share_answer", "share_low", "share_replay",
              "mean_tape_bytes"):
        assert f"train_{k}" in stats and f"val_{k}" in stats
    shares = [stats[f"train_share_{k}"] for k in ("decisions", "answer", "low", "replay")]
    assert abs(sum(shares) - 1.0) < 1e-6
    assert stats["train_replay_seqs"] > 0
    assert stats["train_share_replay"] < 0.15
    assert stats["train_mean_tape_bytes"] > 0


def test_implied_epochs_is_pure_arithmetic():
    assert implied_epochs(100, 32, 1500) == 480.0
    assert implied_epochs(14000, 32, 1500) == 32 * 1500 / 14000


def test_build_shards_val_split_is_deterministic_and_disjoint(tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    kw = dict(episodes=200, seed=3, seq_len=2048, block=32, replay=None, replay_frac=0.0, val_frac=0.1)
    sa = build_shards(a, **kw)
    sb = build_shards(b, **kw)
    va = open(os.path.join(a, "val", "data.bin"), "rb").read()
    vb = open(os.path.join(b, "val", "data.bin"), "rb").read()
    assert va == vb and len(va) > 0                       # même graine → même val
    ta = open(os.path.join(a, "train", "data.bin"), "rb").read()
    seq = 2048
    val_seqs = {va[i:i + seq] for i in range(0, len(va), seq)}
    train_seqs = {ta[i:i + seq] for i in range(0, len(ta), seq)}
    assert not (val_seqs & train_seqs)                    # aucune séquence partagée
    frac = sa["val_seqs"] / (sa["val_seqs"] + sa["train_seqs"])
    assert 0.05 <= frac <= 0.15                           # val_frac 0,1 à ±50 %


def test_tape_storage_is_compact():
    import sys
    from relis.tape.tape import Tape
    from relis.tape import codes as C
    t = Tape()
    for i in range(10_000):
        t.put(i % 256, int(C.Mode.SCAN), W_LOW)
    assert len(t) == 10_000
    assert sys.getsizeof(t.mode) < 20_000
    assert sys.getsizeof(t.wclass) < 20_000 and sys.getsizeof(t.reset) < 20_000
    assert t.mode[5] == int(C.Mode.SCAN) and t.wclass[5] == W_LOW and t.reset[5] == 0
