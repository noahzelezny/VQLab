"""vqlab doctor reports the running interpreter's facts and fails only on blockers."""
from vqlab.agents import doctor


def test_report_names_this_interpreter_and_storage():
    import sys
    r = doctor.report()
    assert r["python"] == sys.executable
    assert set(r["storage"]) == {"scratch", "models", "teachers"}
    assert "mlx" in r["packages"]


def test_missing_root_is_a_problem(monkeypatch, tmp_path):
    monkeypatch.setenv("VQLAB_SCRATCH", str(tmp_path / "gone"))
    from vqlab import config as C
    if hasattr(C._file_paths, "cache_clear"):
        C._file_paths.cache_clear()
    r = doctor.report()
    assert any("scratch root" in p for p in r["problems"])
    assert doctor.main(["--json"]) == 1


def test_stale_scratch_flags_only_old_unclaimed_folders(tmp_path, monkeypatch):
    import os
    import time
    root = tmp_path / "scratch"
    for n in ("night-20260901", "F123-cbprec", "fresh-run", "named-in-log"):
        (root / n).mkdir(parents=True)
    old = time.time() - 30 * 86400
    for n in ("night-20260901", "F123-cbprec", "named-in-log"):
        os.utime(root / n, (old, old))
    log = tmp_path / "FINDINGS-LOG.md"
    log.write_text("F123 codebook precision ... see named-in-log for arrays")
    monkeypatch.setenv("VQLAB_FINDINGS_LOG", str(log))
    stale = [d["name"] for d in doctor.stale_scratch(root)]
    assert stale == ["night-20260901"]          # F123-* claimed, fresh too young, named in log
