"""First run creates the lab folder and config once, and never on a set-up box."""
import io

import pytest

from vqlab import config as C
from vqlab.agents import first_run as F
from vqlab.agents import settings as S


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    for v in (*C._ENV.values(), "HF_HOME", "HF_HUB_CACHE", "EXO_HOME"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.delenv("VQLAB_NO_SETUP", raising=False)
    monkeypatch.setenv("VQLAB_CONFIG", str(tmp_path / "cfg" / "config.toml"))
    monkeypatch.setattr(C, "HOME", tmp_path / ".vqlab")
    monkeypatch.setattr(C, "LAB_DEFAULT", tmp_path / "VQLab")
    monkeypatch.setattr(C, "_XDG_RECORDS", tmp_path / "xdg" / "vqlab")
    from vqlab.agents import model_scan        # never scan the real home's model folders
    monkeypatch.setattr(model_scan, "_home", lambda: tmp_path / "home")
    S._reset()
    yield tmp_path
    S._reset()


def test_first_run_creates_lab_and_config(fresh):
    out = io.StringIO()
    lab = F.run(out)
    assert lab == fresh / "VQLab"
    assert all((lab / d).is_dir() for d in F.SUBFOLDERS)
    assert f'lab = "{lab}"' in (fresh / "cfg" / "config.toml").read_text()
    assert "first run" in out.getvalue()
    assert C.models() == lab / "models"          # the config now names it
    assert F.run(io.StringIO()) is None           # only once


def test_skipped_when_configured_or_env_or_legacy(fresh, monkeypatch):
    monkeypatch.setenv("VQLAB_MODELS_DIR", str(fresh / "m"))
    assert F.run(io.StringIO()) is None
    monkeypatch.delenv("VQLAB_MODELS_DIR")
    (fresh / ".vqlab" / "scratch" / "x").mkdir(parents=True)
    assert F.run(io.StringIO()) is None
    assert not (fresh / "VQLab").exists()


def test_opt_out(fresh, monkeypatch):
    monkeypatch.setenv("VQLAB_NO_SETUP", "1")
    assert F.run(io.StringIO()) is None and not (fresh / "VQLab").exists()


def test_skipped_for_records_from_an_older_install(fresh):
    (fresh / "xdg" / "vqlab" / "registry").mkdir(parents=True)
    before = C.records()
    assert F.run(io.StringIO()) is None
    assert C.records() == before == fresh / "xdg" / "vqlab"
