import json
import os
import torch
from relis.model.config import RelisConfig
from relis.model.relis import RelisModel
from relis.train.checkpoint import save_checkpoint
from relis.infer import cli


def _ckpt(tmp_path):
    cfg = RelisConfig.tiny()
    save_checkpoint(str(tmp_path), RelisModel(cfg), None, None, step=7,
                    cfg_dict=cfg.to_dict(), extra={})
    return str(tmp_path / "last.pt")


def test_load_docs_builds_segments_newest_first(tmp_path):
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.md").write_text("bravo", encoding="utf-8")
    segs = cli.load_docs([str(tmp_path / "a.txt"), str(tmp_path / "b.md")])
    assert [s.content for s in segs] == [b"bravo", b"alpha"]     # dernier joint en premier
    assert "name=b.md" in segs[0].header and "type=doc" in segs[0].header


def test_history_segments_are_newest_first():
    turns = [{"role": "user", "text": "un"}, {"role": "assistant", "text": "deux"},
             {"role": "user", "text": "trois"}]
    segs = cli.history_segments(turns)
    assert [s.content for s in segs] == [b"trois", b"deux", b"un"]
    assert "role=user" in segs[0].header


def test_cli_single_question_writes_conversation(tmp_path, capsys):
    conv = tmp_path / "conv.json"
    cli.main(["--ckpt", _ckpt(tmp_path), "--ask", "bonjour",
              "--conversation", str(conv), "--max_gen", "8", "--no-color"])
    data = json.loads(conv.read_text(encoding="utf-8"))
    assert [t["role"] for t in data["turns"]] == ["user", "assistant"]
    assert data["turns"][0]["text"] == "bonjour"
    assert capsys.readouterr().out.strip() != ""


def test_cli_second_turn_reads_previous_history(tmp_path, capsys):
    conv = tmp_path / "conv.json"
    ck = _ckpt(tmp_path)
    cli.main(["--ckpt", ck, "--ask", "un", "--conversation", str(conv), "--max_gen", "6", "--no-color"])
    cli.main(["--ckpt", ck, "--ask", "deux", "--conversation", str(conv), "--max_gen", "6", "--no-color"])
    data = json.loads(conv.read_text(encoding="utf-8"))
    assert len(data["turns"]) == 4
    assert "history" in capsys.readouterr().out         # le canal a été ouvert au 2e tour


def test_cli_quiet_prints_only_answer(tmp_path, capsys):
    cli.main(["--ckpt", _ckpt(tmp_path), "--ask", "salut", "--max_gen", "6", "--quiet", "--no-color"])
    out = capsys.readouterr().out
    assert "octets lus" not in out
