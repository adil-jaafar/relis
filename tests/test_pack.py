import json, os
import numpy as np
import torch
from relis.data.episodes import iter_episodes
from relis.data.pack import (pad_tape, pack_tapes, replay_tape, TapeShardWriter, TapeWindows,
                             PAD, FLAG_RESET, FLAG_SLOTS, build_shards)
from relis.data.shards import ShardWriter
from relis.tape.tape import build_tape, WEIGHTS


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
        assert all(s["flags"][st] & FLAG_RESET for st in starts)
        # les resets internes (REFRESH) ne portent pas le bit slots
        internal = np.flatnonzero((s["flags"] & FLAG_RESET) & ~(s["flags"] & FLAG_SLOTS))
        assert all(st not in starts for st in internal)


def test_replay_tape_is_full_length_weight_one(tmp_path):
    p = str(tmp_path / "train.bin")
    w = ShardWriter(p); w.add(bytes(range(256)) * 40, "src=t"); w.close()
    import random
    t = replay_tape(p, 512, random.Random(0))
    assert len(t) == 512 and set(t.wclass) == {2} and not any(t.reset)


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
    assert b["w"].dtype == torch.float32 and set(b["w"].unique().tolist()) <= set(WEIGHTS)
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
