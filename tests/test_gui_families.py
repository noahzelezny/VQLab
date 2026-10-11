"""The GUI's Families view reads both places the registry reads."""
import json

from vqlab import config
from vqlab.agents import gui


def test_families_merge_checkout_and_user(tmp_path, monkeypatch):
    co, user = tmp_path / "co", tmp_path / "user"
    (co / "fam_a/teachers/T1").mkdir(parents=True)
    (co / "fam_a/teachers/T1/profile.json").write_text(json.dumps({"x": 1}))
    (user / "fam_a/teachers/T1/research").mkdir(parents=True)
    (user / "fam_a/teachers/T1/research/LEDGER.md").write_text("")
    (user / "fam_b/teachers/T2").mkdir(parents=True)
    monkeypatch.setattr(config, "checkout_families_dir", lambda: co)
    monkeypatch.setattr(gui, "_families_dir", lambda: user)
    fams = {f["family"]: f for f in gui.api_families({})["families"]}
    assert set(fams) == {"fam_a", "fam_b"}
    t1 = fams["fam_a"]["teachers"][0]
    assert t1["profile"] == {"x": 1} and t1["research"] == ["LEDGER.md"]


def test_pick_folder_uses_the_system_chooser(tmp_path, monkeypatch):
    import subprocess
    import sys
    calls = []

    class R:
        returncode, stdout, stderr = 0, str(tmp_path / "pick") + "/\n", ""
    monkeypatch.setattr(subprocess, "run", lambda argv, **k: calls.append(argv) or R())
    for plat, first in (("darwin", "osascript"), ("win32", "powershell")):
        monkeypatch.setattr(sys, "platform", plat)
        calls.clear()
        r = gui.post_pick_folder({"start": str(tmp_path / "missing" / "deeper")})
        assert r == {"ok": True, "path": str(tmp_path / "pick")}
        assert calls[0][0] == first and any(str(tmp_path) in a for a in calls[0])
    monkeypatch.setattr(sys, "platform", "linux")
    calls.clear()
    gui.post_pick_folder({})
    assert calls[-1][0] == sys.executable                     # Tk works anywhere
    R.returncode, R.stdout, R.stderr = 1, "", "execution error: User canceled. (-128)"
    assert gui.post_pick_folder({})["cancelled"]

    def missing(argv, **k):
        raise FileNotFoundError(argv[0])
    monkeypatch.setattr(subprocess, "run", missing)
    r = gui.post_pick_folder({})
    assert not r["ok"] and "type the path" in r["error"]
