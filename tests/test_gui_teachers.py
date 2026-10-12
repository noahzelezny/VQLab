"""agents/teachers.py: the teachers a build can start from, read from
profiles, the fit store and the model scan, never estimated."""
import json
import subprocess

import pytest

from vqlab.agents import model_scan
from vqlab.agents import teachers as T
from vqlab.util.errors import UserError
from vqlab.family import registry


def _geo(d, k, gib, legal=True, fused=True):
    return {"d": d, "K": k, "bits_per_weight": 2.0, "stack_gib": gib, "legal": legal,
            "fused_kernel_ok": fused}


@pytest.fixture
def lab(lab_dirs, monkeypatch):
    tmp_path = lab_dirs.root
    fam = lab_dirs.families
    src = tmp_path / "models" / "Org--Moe"
    src.mkdir(parents=True)
    d = fam / "qwen3_5" / "teachers" / "Org--Moe"
    d.mkdir(parents=True)
    geos = [_geo(4, 256, 1.0), _geo(4, 64, 0.5), _geo(4, 4096, 9.0, fused=False)]
    (d / "profile.json").write_text(json.dumps({
        "created": "2026-01-01", "identity": {"path": str(src), "hf_repo": None},
        "arch": {"kind": "moe", "layers": 4, "vq_layers": [0, 1, 2, 3], "experts_per_token": 8},
        "bytes": {"by_class_gib": {"carry": 1.0, "embed": 1.0, "vq_target": 8.0}, "vq_target_gib_per_bit": 0.5},
        "modules": {"count": 12},
        "geometries": {"gate_proj": geos, "up_proj": geos, "down_proj": geos}}))
    u = fam / "odd" / "teachers" / "X"
    u.mkdir(parents=True)
    (u / "profile.json").write_text(json.dumps({"unknown_family": True}))
    monkeypatch.setattr(T.registry, "family_dirs", lambda: [fam])
    monkeypatch.setattr(T.registry, "teacher_caches", lambda t=None: [{"teacher": "Org--Moe"}])
    monkeypatch.setattr(T, "_fits", lambda: {("qwen3_5", "Org--Moe"): {"d4-K256": 12}})
    monkeypatch.setattr(T, "_PROFILING", set())
    monkeypatch.setattr(T, "_PROFILE_ERRORS", {})
    other = tmp_path / "models" / "Org--Other"
    other.mkdir()
    rows = [{"use": "original", "path": str(src), "name": "Org/Moe", "family": "qwen3_5"},
            {"use": "original", "path": str(other), "name": "Org/Other", "family": "qwen3_5"},
            {"use": "original", "path": str(other), "name": "Org/Whisper", "family": None},
            {"use": "quantized", "path": str(other), "name": "Org/Q", "family": "qwen3_5"}]
    monkeypatch.setattr(model_scan, "load", lambda: rows)
    return tmp_path


def test_teacher_rows(lab):
    out = T.teachers()
    [t] = out["teachers"]                                  # the unknown-family profile is skipped
    assert t["id"] == "qwen3_5/Org--Moe" and t["name"] == "Org/Moe"
    assert t["on_disk"] and t["buildable"] and t["scored"] and t["fits"] == {"d4-K256": 12}
    geos = {g["geo"]: g for g in t["geometries"]}
    assert set(geos) == {"d4-K64", "d4-K256"}             # the kernel-ceiling row is not offered
    assert geos["d4-K256"]["gib"] == 3.0 and geos["d4-K256"]["fit"] is True
    assert geos["d4-K64"]["fit"] is False                 # no measured relerr: not compilable


def test_unprofiled_lists_only_new_family_originals(lab):
    assert [u["name"] for u in T.teachers()["unprofiled"]] == ["Org/Other"]


def _inline_threads(monkeypatch):
    monkeypatch.setattr(T.threading, "Thread", lambda target, daemon: type(
        "Th", (), {"start": lambda self: target()})())


def test_profile_accepts_only_scanned_paths(lab, monkeypatch):
    started = []
    monkeypatch.setattr(T.threading, "Thread", lambda target, daemon: type(
        "Th", (), {"start": lambda self: started.append(target)})())
    for path in ("/etc", ""):
        with pytest.raises(UserError, match="rescan"):
            T.profile({"path": path})
    assert started == []
    r = T.profile({"path": str(lab / "models" / "Org--Other")})
    assert r["ok"] and len(started) == 1
    assert T.profile({"path": str(lab / "models" / "Org--Other")}) == {"ok": True, "running": True}
    assert len(started) == 1                              # already running: not started twice
    assert T.teachers()["unprofiled"][0]["profiling"] is True


@pytest.mark.parametrize("outcome, want", [
    ("rc", "family-profile exited 2: trace | bad header"),
    ("timeout", "family-profile timed out after 30 min"),
    ("launch", "could not run family-profile: no python"),
    ("ok", None)])
def test_a_failed_profile_is_shown_on_its_row(lab, monkeypatch, outcome, want):
    def run_cli(*args, timeout):
        assert args[:2] == ("family-profile", "--teacher")
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(args, timeout)
        if outcome == "launch":
            raise FileNotFoundError("no python")
        return subprocess.CompletedProcess(args, 2 if outcome == "rc" else 0, "", "trace\nbad header\n")
    monkeypatch.setattr(T.qf, "run_cli", run_cli)
    _inline_threads(monkeypatch)
    assert T.profile({"path": str(lab / "models" / "Org--Other")})["ok"]
    [u] = T.teachers()["unprofiled"]
    assert u["profile_error"] == want and u["profiling"] is False


def test_profile_survives_an_unreadable_scan(lab, monkeypatch):
    def broken():
        raise PermissionError("index")
    monkeypatch.setattr(model_scan, "load", broken)
    with pytest.raises(UserError):
        T.profile({"path": str(lab / "models" / "Org--Other")})
    assert T.teachers()["unprofiled"] == []


def test_a_users_profile_overrides_the_checkouts(lab, monkeypatch, tmp_path):
    checkout = tmp_path / "checkout-families"
    d = checkout / "qwen3_5" / "teachers" / "Org--Moe"
    d.mkdir(parents=True)
    (d / "profile.json").write_text(json.dumps({"arch": {"kind": "moe"}, "created": "old"}))
    bad = checkout / "qwen3_5" / "teachers" / "Org--Torn"
    bad.mkdir(parents=True)
    (bad / "profile.json").write_text("{not json")
    monkeypatch.setattr(T.registry, "family_dirs", lambda: [checkout, lab / "families"])
    profs = registry.iter_profiles()
    assert profs[("qwen3_5", "Org--Moe")]["created"] == "2026-01-01"     # the user's wins
    assert ("qwen3_5", "Org--Torn") not in profs                         # unreadable: skipped
    assert registry.teacher_profile("qwen3_5", "Org--Moe")["created"] == "2026-01-01"
    assert registry.teacher_profile("qwen3_5", "Org--Torn") is None
    assert registry.teacher_profile("qwen3_5", "../../etc") is None
    [t] = T.teachers()["teachers"]
    assert t["profiled"] == "2026-01-01"


def test_merge_dedupe_and_buildable(lab, monkeypatch):
    fam = lab / "families"
    copy = lab / "models" / "Copy--Of-Moe"
    copy.mkdir()
    d = fam / "glm5_next" / "teachers" / "Org--Glm"
    d.mkdir(parents=True)
    (d / "profile.json").write_text(json.dumps({"identity": {"path": str(lab / "nowhere")},
                                                "arch": {"kind": "moe"}}))
    rows = model_scan.load() + [
        {"use": "original", "path": str(lab / "models" / "Org--Moe"), "name": "Mirror/Moe",
         "family": "qwen3_5"},                                    # the profiled teacher's own folder
        {"use": "original", "path": str(copy), "name": "Org/Moe", "family": "qwen3_5"},  # its name
        {"use": "original", "path": str(lab / "models" / "Org--Other"), "name": "Again/Other",
         "family": "qwen3_5"}]                                   # one folder, listed once
    monkeypatch.setattr(model_scan, "load", lambda: rows)
    out = T.teachers()
    by = {t["id"]: t for t in out["teachers"]}
    assert set(by) == {"qwen3_5/Org--Moe", "glm5_next/Org--Glm"}
    assert by["glm5_next/Org--Glm"]["buildable"] is False and by["glm5_next/Org--Glm"]["on_disk"] is False
    assert [u["name"] for u in out["unprofiled"]] == ["Org/Other"]
