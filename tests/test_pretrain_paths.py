import os
import pytest
from relis.train.pretrain import resolve_data_path


def test_existing_path_is_returned_unchanged(tmp_path):
    p = tmp_path / "train.bin"
    p.write_bytes(b"x")
    assert resolve_data_path(str(p), search_roots=[str(tmp_path)]) == str(p)


def test_missing_path_is_found_by_basename_under_roots(tmp_path):
    real = tmp_path / "datasets" / "user" / "relis-shards" / "train.bin"
    real.parent.mkdir(parents=True)
    real.write_bytes(b"x")
    got = resolve_data_path("/kaggle/input/relis-shards/train.bin", search_roots=[str(tmp_path)])
    assert os.path.samefile(got, real)


def test_missing_path_without_match_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_data_path(str(tmp_path / "absent" / "train.bin"), search_roots=[str(tmp_path)])


def test_ambiguous_match_raises(tmp_path):
    for d in ("a", "b"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "train.bin").write_bytes(b"x")
    with pytest.raises(FileNotFoundError):
        resolve_data_path(str(tmp_path / "absent" / "train.bin"), search_roots=[str(tmp_path)])
