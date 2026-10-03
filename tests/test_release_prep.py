"""release-prep compares small files to the Hub by git blob hash; it must be
exactly git's (what the Hub reports as blob_id)."""
import subprocess

from vqlab.ship import release_prep as R


def test_git_blob_matches_git(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"hello\nworld\n")
    want = subprocess.run(["git", "hash-object", str(p)], capture_output=True, text=True).stdout.strip()
    assert R._git_blob(p) == want


def test_junk(tmp_path):
    (tmp_path / ".DS_Store").write_text("x")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "model.py").write_text("x")
    names = {p.rsplit("/", 1)[-1] for p in R._junk(tmp_path)}
    assert names == {".DS_Store", "__pycache__"}
