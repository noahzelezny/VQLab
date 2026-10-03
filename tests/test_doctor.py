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
