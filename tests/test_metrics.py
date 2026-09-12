import torch
from relis.model.config import RelisConfig
from relis.model.relis import RelisModel
from relis.train.metrics import balanced_accuracy, format_decision_report, decision_accuracy

# Chiffres observés sur un run réel (spec §9) : READ et CONT dominent (majorité
# toujours correcte), STOP/END/SKIP/NOTE/NEXT restent à zéro.
_OBSERVED_PER_CODE = {
    "READ": [406, 406], "CONT": [362, 362], "REFRESH": [13, 16],
    "STOP": [0, 54], "END": [0, 38], "SKIP": [0, 21], "NOTE": [0, 16], "NEXT": [0, 11],
}


def test_balanced_accuracy_penalises_majority_collapse():
    per_code = _OBSERVED_PER_CODE
    acc_balanced = balanced_accuracy(per_code)
    correct = sum(c for c, _ in per_code.values())
    total = sum(n for _, n in per_code.values())
    acc = correct / total
    assert abs(acc_balanced - 0.3516) < 1e-3
    assert abs(acc - 0.8452) < 1e-3


def test_balanced_accuracy_ignores_absent_codes():
    import math
    with_absent = dict(_OBSERVED_PER_CODE, GHOST=[0, 0])
    assert balanced_accuracy(with_absent) == balanced_accuracy(_OBSERVED_PER_CODE)
    assert math.isnan(balanced_accuracy({"A": [0, 0], "B": [0, 0]}))
    assert math.isnan(balanced_accuracy({}))


def test_format_decision_report_orders_failures_first():
    res = {"acc": 0.8452380952380952, "n": 924, "per_code": _OBSERVED_PER_CODE,
           "acc_balanced": balanced_accuracy(_OBSERVED_PER_CODE)}
    s = format_decision_report(res)
    assert s.startswith("équilibrée")
    assert s.index("STOP 0/54") < s.index("READ 406/406")


def test_decision_accuracy_returns_acc_balanced_end_to_end(tmp_path):
    import math
    from relis.data.pack import build_shards, TapeWindows
    from relis.data.shards import ShardWriter
    pre = str(tmp_path / "pre.bin")
    w = ShardWriter(pre); w.add(("le chat dort. " * 400).encode(), "src=t"); w.close()
    out = str(tmp_path / "tapes")
    build_shards(out, episodes=60, seed=0, seq_len=1024, block=32, replay=pre,
                 replay_frac=0.2, val_frac=0.1)
    va = TapeWindows(str(tmp_path / "tapes" / "val"))
    m = RelisModel(RelisConfig.tiny())
    r = decision_accuracy(m, va, n_batches=1, batch_size=2, device="cpu")
    assert "acc_balanced" in r
    assert (0.0 <= r["acc_balanced"] <= 1.0) or math.isnan(r["acc_balanced"])
