"""The `records` location, `vqlab config set` validation, and the opt-in
Knurlogic architecture hook. Every path here is a pytest temp dir."""
import os
import pathlib
import subprocess
import sys

import pytest

from vqlab import config as C
from vqlab.agents import config_cmd
from vqlab.agents import settings as S

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
EXTRA_ENV = ("VQLAB_FINDINGS_LOG", "VQLAB_FAMILIES_DIR", "VQLAB_QUEUE_DIR", "XDG_DATA_HOME")


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    for v in (*C._ENV.values(), *EXTRA_ENV):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("VQLAB_CONFIG", str(tmp_path / "c.toml"))
    S._reset()
    yield tmp_path / "c.toml"
    S._reset()


def test_records_accessors_default_and_env(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    # the default is computed at import; the env var is read per call
    monkeypatch.setitem(C._DEFAULT, "records", tmp_path / "xdg" / "vqlab")
    r = C.records()
    assert r == tmp_path / "xdg" / "vqlab"
    assert C.findings_log() == r / "FINDINGS-LOG.md"
    assert C.registry_dir() == r / "registry"
    assert C.families_dir() == r / "families"
    assert C.research_dir() == r / "research"
    assert C.queues_dir() == r / "queues"
    monkeypatch.setenv("VQLAB_RECORDS", str(tmp_path / "rec"))
    assert C.records() == tmp_path / "rec"
    assert C.registry_dir() == tmp_path / "rec" / "registry"
    monkeypatch.setenv("VQLAB_FINDINGS_LOG", str(tmp_path / "elsewhere.md"))
    assert C.findings_log() == tmp_path / "elsewhere.md"
    monkeypatch.setenv("VQLAB_FAMILIES_DIR", str(tmp_path / "fam"))
    assert C.families_dir() == tmp_path / "fam"


def test_records_default_is_outside_package():
    pkg = pathlib.Path(C.__file__).resolve().parent
    d = C._DEFAULT["records"].resolve()
    assert pkg not in d.parents and d != pkg
    assert SRC.parent not in d.parents


def test_records_from_config_file(cfg, tmp_path):
    cfg.write_text(f'[paths]\nrecords = "{tmp_path / "lab"}"\n')
    S._reset()
    assert C.records() == tmp_path / "lab"
    assert C.findings_log() == tmp_path / "lab" / "FINDINGS-LOG.md"
    assert S.source("records") == str(cfg)


def test_validate(cfg, tmp_path):
    assert not S.validate("nope", str(tmp_path))["ok"]
    assert not S.validate("scratch", "relative/dir")["ok"]
    missing = tmp_path / "a" / "b"
    r = S.validate("scratch", str(missing))
    assert not r["ok"] and "does not exist" in r["error"] and not missing.exists()
    r = S.validate("scratch", str(missing), create=True)
    assert r["ok"] and r["created"] and missing.is_dir() and r["free_bytes"] > 0
    f = tmp_path / "file"
    f.write_text("x")
    assert "not a directory" in S.validate("models", str(f))["error"]
    # a file key checks its parent directory
    assert S.validate("gpu_lease", str(tmp_path / "gpu.lease"))["ok"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes anywhere")
def test_validate_unwritable(cfg, tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        r = S.validate("scratch", str(ro))
        assert not r["ok"] and "not writable" in r["error"]
    finally:
        ro.chmod(0o700)


def test_set_preserves_other_content(cfg, tmp_path):
    cfg.write_text('# my notes\n[paths]\nscratch = "/old/scratch"  # keep?\n'
                   'fit_store = [\n  "/old/fits",\n]\n\n[boxes.b]\nssh = "u@h"\n')
    S._reset()
    new = tmp_path / "s"
    new.mkdir()
    r = S.set_value("scratch", str(new))
    assert r["ok"] and r["config"] == str(cfg)
    assert C.scratch() == new
    r = S.set_value("fit_store", str(new))
    assert r["ok"] and C.fit_store() == [new]
    r = S.set_value("records", str(tmp_path / "rec"), create=True)
    assert r["ok"] and C.records() == tmp_path / "rec"
    text = cfg.read_text()
    assert text.startswith("# my notes\n") and '[boxes.b]\nssh = "u@h"' in text
    assert "/old/" not in text
    assert C.boxes()["b"]["ssh"] == "u@h"


def test_set_creates_file_and_refuses_bad(cfg, tmp_path, capsys):
    assert config_cmd.main(["set", "models", str(tmp_path / "missing")]) == 2
    assert not cfg.exists()
    assert config_cmd.main(["set", "models", str(tmp_path / "m"), "--create"]) == 0
    assert C.models() == tmp_path / "m"
    assert "models = " in capsys.readouterr().out


def test_set_reports_env_shadow(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("VQLAB_SCRATCH", str(tmp_path))
    r = S.set_value("scratch", str(tmp_path))
    assert r["ok"] and r["shadowed_by"] == "$VQLAB_SCRATCH"


def test_locations_and_campaign_sizes(cfg, tmp_path, monkeypatch):
    sc = tmp_path / "scratch"
    (sc / "F1-x").mkdir(parents=True)
    (sc / "F1-x" / "blob").write_bytes(b"\0" * 8192)
    (sc / ".hidden").mkdir()
    monkeypatch.setenv("VQLAB_SCRATCH", str(sc))
    keys = [r["key"] for r in S.locations()]
    assert keys == list(C.KEYS)
    rows = S.campaign_sizes()
    assert [r["name"] for r in rows] == ["F1-x"] and rows[0]["bytes"] >= 8192
    assert S.campaign_sizes(budget=0)[0]["bytes"] is None   # out of time -> None, not a hang


def test_gui_post_settings(cfg, tmp_path):
    from vqlab.agents import gui
    r = gui.post_settings({"key": "scratch", "path": str(tmp_path / "g"), "create": True})
    assert r["ok"] and C.scratch() == tmp_path / "g"
    assert not gui.post_settings({"key": "scratch", "path": ""})["ok"]
    s = gui.api_settings({})
    assert {x["key"] for x in s["locations"]} == set(C.KEYS)


def _py(code):
    env = {k: v for k, v in os.environ.items() if k != "VQLAB_VENDORED_ARCH"}
    env["PYTHONPATH"] = str(SRC)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=env, timeout=120)


def test_import_leaves_meta_path_unchanged():
    # vqlab's own stage-alias finder (_layout._Alias: old vqlab.* names only)
    # is internal; nothing that resolves another package's modules may appear
    r = _py("import sys; before = list(sys.meta_path); import vqlab; "
            "from vqlab import _layout; from vqlab.core import arch_finder as arch; "
            "new = [f for f in sys.meta_path if f not in before "
            "and not isinstance(f, _layout._Alias)]; "
            "print(new == [] and arch._FINDER not in sys.meta_path)")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1] == "True"


def test_use_knurlogic_arch_installs_one_finder(monkeypatch):
    import vqlab
    from vqlab.core.box import arch_finder as arch

    class _Reg:
        def available(self):
            return []
    monkeypatch.setattr(arch, "_register", lambda: _Reg())
    monkeypatch.delenv("VQLAB_VENDORED_ARCH", raising=False)
    arch.uninstall()
    before = list(sys.meta_path)
    try:
        assert vqlab.use_knurlogic_arch() is True
        assert vqlab.use_knurlogic_arch() is True             # idempotent
        added = [f for f in sys.meta_path if f not in before]
        assert added == [arch._FINDER]
        arch.uninstall()
        monkeypatch.setenv("VQLAB_VENDORED_ARCH", "0")
        assert vqlab.use_knurlogic_arch() is False
        assert arch._FINDER not in sys.meta_path
    finally:
        arch.uninstall()
        if arch._FINDER in before:
            sys.meta_path.insert(before.index(arch._FINDER), arch._FINDER)
