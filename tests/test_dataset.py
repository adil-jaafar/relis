import torch
from relis.data.shards import ShardWriter
from relis.data.dataset import ByteWindows


def test_byte_windows_shift_and_range(tmp_path):
    p = str(tmp_path / "train.bin")
    w = ShardWriter(p)
    w.add(bytes(range(256)) * 20, "src=t")
    w.close()
    ds = ByteWindows(p, seq_len=16)
    g = torch.Generator().manual_seed(0)
    x, y = ds.sample(4, g)
    assert x.shape == (4, 16) and y.shape == (4, 16)
    assert x.dtype == torch.long
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert x.min() >= 0 and x.max() <= 255
    assert len(ds) == w.total_bytes
