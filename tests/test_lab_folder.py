"""The lab folder: every location defaults under one visible folder, and a
setup from before it keeps resolving where it did."""
import pytest

from vqlab import config as C
from vqlab.agents import settings as S


@pytest.fixture
def home(tmp_path, monkeypatch):
    for v in C._ENV.values():
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("VQLAB_CONFIG", str(tmp_path / "c.toml"))
    monkeypatch.setattr(C, "HOME", tmp_path / ".vqlab")
    monkeypatch.setattr(C, "LAB_DEFAULT", tmp_path / "VQLab")
    monkeypatch.setattr(C, "_XDG_RECORDS", tmp_path / "xdg" / "vqlab")
    S._reset()
    yield tmp_path
    S._reset()


def test_fresh_install_puts_everything_under_the_lab_folder(home):
    lab = home / "VQLab"
    assert C.lab() == lab
    assert (C.scratch(), C.models(), C.teachers(), C.fit_store(), C.records()) == \
        (lab / "scratch", lab / "models", lab / "models", [lab / "fits"], lab / "records")


def test_setting_lab_moves_every_default(home, monkeypatch):
    monkeypatch.setenv("VQLAB_HOME", str(home / "big"))
    assert C.models() == home / "big" / "models" and C.records() == home / "big" / "records"
    monkeypatch.setenv("VQLAB_MODELS_DIR", str(home / "fast"))
    assert C.models() == home / "fast" and C.scratch() == home / "big" / "scratch"


def test_lab_from_config_file(home):
    (home / "c.toml").write_text(f'[paths]\nlab = "{home / "L"}"\n')
    S._reset()
    assert C.lab() == home / "L" and C.teachers() == home / "L" / "models"
    assert S.source("teachers") == "same as models"
    rows = {r["key"]: r for r in S.locations()}
    assert rows["teachers"]["shared_with"] == "models" and rows["models"]["shared_with"] is None


def test_pre_lab_setups_keep_their_data(home):
    (home / ".vqlab" / "models" / "m").mkdir(parents=True)
    assert C.teachers() == home / ".vqlab" / "teachers"
    (home / "xdg" / "vqlab").mkdir(parents=True)
    assert C.models() == home / ".vqlab" / "models"
    assert C.scratch() == home / ".vqlab" / "scratch"
    assert C.records() == home / "xdg" / "vqlab"


def test_usage_groups_locations_by_volume(home, monkeypatch):
    monkeypatch.setenv("VQLAB_HOME", str(home / "VQLab"))
    (home / "VQLab" / "models").mkdir(parents=True)
    (home / "VQLab" / "models" / "w.bin").write_bytes(b"x" * 8192)
    u = S.usage()
    assert u["lab"] == str(home / "VQLab")
    locs = {l["key"]: l for v in u["volumes"] for l in v["locations"]}
    assert set(locs) == {"models", "scratch", "fit_store", "records"}   # teachers share models/
    assert locs["models"]["bytes"] >= 8192
    monkeypatch.setenv("VQLAB_TEACHERS_DIR", str(home / "ext"))
    locs = {l["key"]: l for v in S.usage()["volumes"] for l in v["locations"]}
    assert locs["teachers"]["label"] == "bf16 teachers" and locs["teachers"]["bytes"] == 0
    assert all({"mount", "total", "free", "system"} <= set(v) for v in u["volumes"])


def test_records_name_shared_teachers_as_models(home, monkeypatch):
    """bf16 teachers in the models folder are recorded as <models>/..., so a
    record stays right after teachers are set apart."""
    monkeypatch.setenv("VQLAB_HOME", str(home / "L"))
    art = home / "L" / "models" / "Art-1"
    art.mkdir(parents=True)
    assert C.portable(art) == "<models>/Art-1"
    assert dict((lab, p) for p, lab in C.private_prefixes()).get("<models>")
    assert "<teachers>" not in {lab for _, lab in C.private_prefixes()}
    monkeypatch.setenv("VQLAB_TEACHERS_DIR", str(home / "ext"))
    (home / "ext" / "T").mkdir(parents=True)
    assert C.portable(home / "ext" / "T") == "<teachers>/T"
    assert C.portable(art) == "<models>/Art-1"
