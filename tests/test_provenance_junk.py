"""Finder's .DS_Store is not a build output: it must neither enter a record
nor break one when it appears (or changes) after the build."""
from vqlab.records import provenance as p


def test_ds_store_neither_recorded_nor_breaks_verify(tmp_path):
    (tmp_path / "model.py").write_text("x = 1\n")
    (tmp_path / ".DS_Store").write_bytes(b"one")
    rec = p.write_build_record(tmp_path, tool="test", argv=["test"])
    assert ".DS_Store" not in rec["outputs"]
    (tmp_path / ".DS_Store").write_bytes(b"two, after the build")
    (tmp_path / "._model.py").write_bytes(b"appledouble")
    assert p.verify(tmp_path, p.load(tmp_path)) == []


def test_a_real_change_still_fails_verify(tmp_path):
    (tmp_path / "model.py").write_text("x = 1\n")
    p.write_build_record(tmp_path, tool="test", argv=["test"])
    (tmp_path / "model.py").write_text("x = 2\n")
    assert [n for n, _ in p.verify(tmp_path, p.load(tmp_path))] == ["model.py"]
