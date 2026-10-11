"""Records moved under `records` (2026-10): warn about data left at the old
places, and refuse to restart F-numbering. Temp HOME, temp checkout, temp
config; nothing real is read or written."""
import pytest

from vqlab import config as C
from vqlab.agents import config_cmd, doctor
from vqlab.agents import settings as S
from vqlab.ops import findings as F
from vqlab.ops.errors import LabError


@pytest.fixture
def env(tmp_path, monkeypatch):
    for v in (*C._ENV.values(), "VQLAB_FINDINGS_LOG", "VQLAB_QUEUE_DIR", "XDG_DATA_HOME"):
        monkeypatch.delenv(v, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("VQLAB_CONFIG", str(tmp_path / "c.toml"))
    monkeypatch.setitem(C._DEFAULT, "records", home / ".local" / "share" / "vqlab")
    co = tmp_path / "checkout"
    co.mkdir()
    (co / "pyproject.toml").write_text("")
    monkeypatch.setattr(C, "CHECKOUT", co)
    monkeypatch.setattr(F, "FINDINGS_LOG", C.findings_log())
    monkeypatch.setattr(F, "_FLOCK_DIR", tmp_path / ".f-locks")
    S._reset()
    yield {"home": home, "co": co, "tmp": tmp_path}
    S._reset()


def _legacy(e):
    (e["co"] / "lab").mkdir()
    (e["co"] / "lab" / "FINDINGS-LOG.md").write_text("## F7 x\n")
    (e["co"] / "registry").mkdir()
    (e["co"] / "registry" / "artifacts.jsonl").write_text("{}\n")
    (e["co"] / "registry" / "README.md").write_text("x")
    q = e["home"] / ".vqlab" / "queues" / "20260101-a"
    q.mkdir(parents=True)


def test_clean_install_no_warnings(env):
    assert C.legacy_records() == []
    assert F.next_f_number()["next"] == 1          # a genuinely new lab starts at F1


def test_legacy_data_warns_everywhere(env, capsys):
    _legacy(env)
    whats = {r["what"] for r in C.legacy_records()}
    assert whats == {"findings log", "artifact registry", "queue records"}
    fixes = " ".join(r["fix"] for r in C.legacy_records())
    assert f"vqlab lab config set records {env['co'] / 'lab'}" in fixes
    assert f"VQLAB_QUEUE_DIR={env['home'] / '.vqlab' / 'queues'}" in fixes
    config_cmd.show()
    assert capsys.readouterr().out.count("WARNING: records moved") == 3
    notes = [n for n in doctor.report()["notes"] if "records moved" in n]
    assert len(notes) == 3


def test_registry_readme_only_is_not_legacy(env):
    (env["co"] / "registry").mkdir()
    (env["co"] / "registry" / "README.md").write_text("x")
    (env["co"] / "registry" / "empty.jsonl").write_text("")
    assert C.legacy_records() == []


def test_records_set_silences(env, monkeypatch):
    _legacy(env)
    monkeypatch.setenv("VQLAB_RECORDS", str(env["tmp"] / "rec"))
    assert C.legacy_records() == []


def test_findings_refuse_f1_when_log_moved(env):
    _legacy(env)
    for call in (F.next_f_number, lambda: F.next_f_number(reserve=True), F.findings_tail,
                 lambda: F.findings_append("h", "a", "i", "p", "m", "NULL")):
        with pytest.raises(LabError) as ei:
            call()
        assert ei.value.code == "RECORDS_MOVED"
        assert str(env["co"] / "lab" / "FINDINGS-LOG.md") in ei.value.message
    assert not (env["tmp"] / ".f-locks").exists()


def test_findings_ok_when_pointed_at_legacy(env, monkeypatch):
    _legacy(env)
    log = env["co"] / "lab" / "FINDINGS-LOG.md"
    monkeypatch.setenv("VQLAB_FINDINGS_LOG", str(log))
    monkeypatch.setattr(F, "FINDINGS_LOG", C.findings_log())
    assert F.next_f_number()["next"] == 8
