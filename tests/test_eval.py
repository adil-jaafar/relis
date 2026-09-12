import random
import torch
from relis.model.config import RelisConfig
from relis.model.relis import RelisModel
from relis.train.checkpoint import save_checkpoint
from relis.data.episodes import generate_case
from relis.infer.controller import Budget, Controller
from relis.eval.harness import judge, load_controller, run_case


def _ctl():
    return Controller(RelisModel(RelisConfig.tiny()).eval(), "cpu",
                      Budget(max_gen_bytes=24, max_read_bytes=4000))


def test_run_case_reports_counters_and_stop():
    c = generate_case(random.Random(1), "fact_recall")
    r = run_case(_ctl(), c)
    assert set(r) >= {"answer", "read", "written", "saved", "stop_at", "read_segments"}
    assert r["written"] <= 24
    assert r["stop_at"] is None or isinstance(r["stop_at"], int)


def test_judge_absent_and_value():
    c = generate_case(random.Random(2), "absent")
    assert judge(c, "Je ne trouve pas cette information.") is True
    assert judge(c, "C'est 42.") is False
    f = generate_case(random.Random(3), "fact_recall")
    assert judge(f, f"La réponse est {f.answer_value} voilà") is True
    assert judge(f, "aucune idée") is False


def test_load_controller_from_local_checkpoint(tmp_path):
    cfg = RelisConfig.tiny()
    save_checkpoint(str(tmp_path), RelisModel(cfg), None, None, 11, cfg.to_dict(), {})
    ctl, step = load_controller(str(tmp_path / "last.pt"), "cpu")
    assert step == 11 and isinstance(ctl, Controller)
