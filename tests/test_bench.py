from relis.model.config import RelisConfig
from relis.model.relis import RelisModel
from relis.train.bench import throughput


def test_throughput_positive_cpu():
    m = RelisModel(RelisConfig.tiny())
    bps = throughput(m, seq_len=64, batch_size=2, device="cpu", amp=False, steps=2)
    assert bps > 0
