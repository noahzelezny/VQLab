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
