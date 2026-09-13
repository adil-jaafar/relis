from relis.eval import distractor
from relis.eval.harness import load_controller
from relis.model.config import RelisConfig
from relis.model.relis import RelisModel
from relis.train.checkpoint import save_checkpoint


def test_distractor_probe_runs_four_cases(tmp_path):
    cfg = RelisConfig.tiny()
    save_checkpoint(str(tmp_path), RelisModel(cfg), None, None, step=7, cfg_dict=cfg.to_dict(), extra={})
    ctl, _ = load_controller(str(tmp_path / "last.pt"), "cpu")
    rows = distractor.evaluate(ctl, max_gen=6)
    assert len(rows) == 4
    assert [r["kind"][0] for r in rows] == ["A", "B", "C", "D"]
    report = distractor.format_report(rows)
    assert report.count("attendu") == 4 and "stop_at=" in report


def test_distractor_cases_are_well_formed():
    for c in distractor.cases():
        assert c.history[0].header.startswith("role=")
        assert c.expect
