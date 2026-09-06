import os
from relis.data.prepare import build_shards


def _fake(name):
    for i in range(50):
        yield (f"document {name} numéro {i} ".encode() * 20, f"src={name};i={i}")


def test_build_shards_respects_budget_and_split(tmp_path):
    out = str(tmp_path)
    stats = build_shards(["a", "b"], out, {"a": 20_000, "b": 5_000}, val_fraction=0.1, iterator_factory=_fake)
    assert os.path.exists(os.path.join(out, "train.bin"))
    assert os.path.exists(os.path.join(out, "val.bin"))
    assert stats["a"] >= 20_000 and stats["a"] < 20_000 + 2_000
    assert stats["b"] >= 5_000 and stats["b"] < 5_000 + 2_000
    assert stats["val_bytes"] > 0 and stats["train_bytes"] > stats["val_bytes"]
