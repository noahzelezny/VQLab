"""check-bundle STALE holds only at the flags a revision was CERTIFIED at
(equivalent_revisions.json `flags`, legacy 4096/off/off when absent): the
code match is modulo VQ_* defaults, so the baked values are compared too.
Synthetic bundles from the registry's own revision texts."""
import json

import pytest

from vqlab._layout import runtime_file
from vqlab.gate import check_bundle as cb

REG = runtime_file("equivalent_revisions.json")
REVS = json.loads(REG.read_text())["revisions"]
FLAGGED = next(r for r in REVS if r.get("flags"))
LEGACY = next(r for r in REVS if not r.get("flags") and r["path"].endswith("vq_switch.py"))


def _rp():
    import importlib.util as u
    sp = u.spec_from_file_location("runtime_profile", runtime_file("runtime_profile.py"))
    m = u.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def _bundle(tmp_path, rev, flags=None):
    text = (REG.parent / rev["file"]).read_text()
    if flags:
        text = _rp().apply_flags(text, flags)
    art = tmp_path / "art"
    art.mkdir(parents=True)
    (art / "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe",
                                                 "vq_modules": {}}))
    (art / "model.py").write_text(text + "\n# loader shim\n")
    return art


def test_flagged_revision_at_its_flags_is_stale(tmp_path):
    rc, lines = cb.check(_bundle(tmp_path, FLAGGED))
    out = "\n".join(lines)
    assert rc == 0 and out.startswith("STALE"), out
    assert "VQ_FUSED_MAX_N=512, VQ_GEMMSEG_ACCS=1, VQ_GEMMSEG_CVEC=1" in out
    assert cb.check(_bundle(tmp_path / "s", FLAGGED), strict=True)[0] == 2


def test_flagged_revision_at_other_flags_is_not_stale(tmp_path):
    art = _bundle(tmp_path, FLAGGED, {"VQ_FUSED_MAX_N": "4096", "VQ_GEMMSEG_ACCS": "0",
                                      "VQ_GEMMSEG_CVEC": "0"})
    rc, lines = cb.check(art)
    out = "\n".join(lines)
    assert rc == 1 and not out.startswith("STALE") and "FAIL" in out, out
    assert "not STALE" in out
    assert "VQ_FUSED_MAX_N" in out and "bundle=4096  certified=512" in out
    assert "bundle=False  certified=True" in out


def test_flagged_revision_one_flag_off(tmp_path):
    rc, lines = cb.check(_bundle(tmp_path, FLAGGED, {"VQ_GEMMSEG_CVEC": "0"}))
    out = "\n".join(lines)
    assert rc == 1 and "VQ_GEMMSEG_CVEC" in out and "VQ_FUSED_MAX_N " not in out, out


def test_legacy_revision_is_stale_at_legacy_flags(tmp_path):
    rc, lines = cb.check(_bundle(tmp_path, LEGACY))
    out = "\n".join(lines)
    assert rc == 0 and out.startswith("STALE") and "LEGACY" in out, out
    assert "VQ_FUSED_MAX_N=4096, VQ_GEMMSEG_ACCS=0, VQ_GEMMSEG_CVEC=0" in out


@pytest.mark.parametrize("raw,want", [("1", True), ("0", False), ("true", True), ("", False)])
def test_baked_bool_parse(raw, want):
    text = f'os.environ.get("VQ_GEMMSEG_ACCS", "{raw}")'
    assert cb._baked_prefill_flags(text, _rp())["VQ_GEMMSEG_ACCS"] is want


def test_verdict_words(tmp_path):
    """verdict(): the flagged revision gains only for vq_skipzero, so a
    non-skipzero bundle on it is CURRENT (a rebundle changes nothing); the
    legacy revision gains for everyone, so it stays STALE."""
    assert FLAGGED.get("gains_only_for") == ["vq_skipzero"]
    v = cb.verdict(_bundle(tmp_path / "a", FLAGGED))
    assert v["verdict"] == "CURRENT" and v["note"] == FLAGGED["gains"]
    assert cb.verdict(_bundle(tmp_path / "b", LEGACY))["verdict"] == "STALE"
    bad = _bundle(tmp_path / "c", FLAGGED, {"VQ_GEMMSEG_CVEC": "0"})
    assert cb.verdict(bad)["verdict"] == "FAIL"
    assert cb.verdict(tmp_path / "nothing")["verdict"] == "MISSING"
