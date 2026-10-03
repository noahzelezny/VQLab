"""The numerics build stamp (F194): what it records, what it refuses, and the
readers that consume it (kl-ladder preflight, kl-pair, card-tables,
runtime-equiv). CPU only: no model load, no mx array."""
import json
import sys
import types

import numpy as np
import pytest
from safetensors.numpy import save_file

from vqlab.core import numerics as N

RUN = {"mlx": "0.31.2", "mlx_lm": "0.31.9", "arch_sha256": "aa", "scorer_variant": "shared-clamp"}


@pytest.fixture
def toy_arch(tmp_path, monkeypatch):
    f = tmp_path / "toyarch_q.py"
    f.write_text("# toy architecture\n")
    mod = types.ModuleType("mlx_lm.models.toyarch_q")
    mod.__file__ = str(f)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.toyarch_q", mod)
    return f


def test_build_stamps_versions_arch_and_variant(toy_arch, monkeypatch):
    monkeypatch.setattr(N, "versions", lambda: {"mlx": "X", "mlx_lm": "Y"})
    b = N.build("toyarch_q", variant="v1")
    assert (b["mlx"], b["mlx_lm"], b["scorer_variant"]) == ("X", "Y", "v1")
    assert b["arch_file"] == str(toy_arch)
    assert b["arch_sha256"] == N._sha(toy_arch)
    # no plugin for this model_type -> no variant
    assert N.build("toyarch_q")["scorer_variant"] is None


def test_build_records_loaded_file_without_comparing_it(toy_arch, tmp_path, monkeypatch):
    bundle = tmp_path / "model.py"
    bundle.write_text("# bundled VQ runtime\n")
    mod = types.ModuleType("toy_bundle_mod")
    mod.__file__ = str(bundle)
    monkeypatch.setitem(sys.modules, "toy_bundle_mod", mod)
    Model = type("Model", (), {"__module__": "toy_bundle_mod"})
    b = N.build("toyarch_q", model=Model(), variant=None)
    assert b["loaded_file"] == str(bundle) and b["arch_file"] == str(toy_arch)
    assert "loaded_sha256" not in N.COMPARED


def test_deepseek_variant_via_plugin_hook(monkeypatch):
    from vqlab.family import variant_for
    monkeypatch.delenv("VQLAB_DS4_SHARED_CLAMP", raising=False)
    assert variant_for("deepseek_v4") == "shared-clamp"
    monkeypatch.setenv("VQLAB_DS4_SHARED_CLAMP", "0")
    assert variant_for("deepseek_v4") is None
    assert variant_for("no_such_family") is None


def test_check_cache_match_refuse_allow():
    assert N.check_cache({"numerics": dict(RUN)}, RUN)["status"] == "match"
    other = {**RUN, "mlx": "0.32.0.dev"}
    with pytest.raises(SystemExit, match="mlx: cache '0.32.0.dev' vs run '0.31.2'"):
        N.check_cache({"numerics": other}, RUN)
    rec = N.check_cache({"numerics": other}, RUN, allow=True)
    assert rec["status"] == "mismatch-allowed"
    assert rec["mismatches"] == [["mlx", "0.32.0.dev", "0.31.2"]]
    with pytest.raises(SystemExit, match="arch_sha256"):
        N.check_cache({"numerics": {**RUN, "arch_sha256": "bb"}}, RUN)


def test_check_cache_unstamped_warns_but_still_checks_variant(capsys):
    rec = N.check_cache({"scorer_variant": "shared-clamp"}, RUN)
    assert rec["status"] == "unstamped-cache"
    assert "NO numerics build stamp" in capsys.readouterr().err
    with pytest.raises(SystemExit, match="scorer_variant"):
        N.check_cache({}, RUN)                      # pre-F195 cache: variant None


def _sidecar_rec(tokens="t1", build=RUN, n=4):
    return {"kl_positions": n, "numerics": dict(build),
            "kl_cache": {"tokens_sha256": tokens, "teacher_head_sha256": "h", "teacher_bytes": 9}}


def test_pairing_problems():
    assert N.pairing_problems(_sidecar_rec(), _sidecar_rec()) == []
    assert any("different cache" in p for p in N.pairing_problems(_sidecar_rec(), _sidecar_rec("t2")))
    assert any("different build" in p for p in
               N.pairing_problems(_sidecar_rec(), _sidecar_rec(build={**RUN, "mlx_lm": "0.30"})))
    assert N.pairing_problems(None, _sidecar_rec())
    assert N.pairing_problems({"kl_positions": 4}, _sidecar_rec())    # old sidecar


def test_cache_identity(tmp_path):
    (tmp_path / "tokens.safetensors").write_bytes(b"tok")
    (tmp_path / "teacher_topk.safetensors").write_bytes(b"lp" * 10)
    (tmp_path / "meta.json").write_text(json.dumps({"numerics": RUN}))
    ci = N.cache_identity(tmp_path)
    assert ci["teacher_file"] == "teacher_topk.safetensors" and ci["teacher_bytes"] == 20
    assert ci["numerics"] == RUN and len(ci["tokens_sha256"]) == 64


def _arm(d, name, vals, rec):
    f = d / f"{name}__prose.safetensors"
    save_file({"kl_millinats": np.array(vals, dtype=np.float32)}, str(f))
    if rec is not None:
        (d / (f.name + ".json")).write_text(json.dumps(rec))


def test_kl_pair_verifies_from_sidecars(tmp_path, monkeypatch, capsys):
    from vqlab.score import kl_pair
    argv = ["kl-pair", "--per-pos-dir", str(tmp_path), "--arm", "x", "--ref", "r", "--corpus", "prose"]
    _arm(tmp_path, "r", [1, 2, 3, 4], _sidecar_rec())
    _arm(tmp_path, "x", [1, 2, 3, 5], _sidecar_rec())
    monkeypatch.setattr(sys, "argv", argv)
    assert kl_pair.main() == 0
    assert "pairing verified" in capsys.readouterr().out
    _arm(tmp_path, "x", [1, 2, 3, 5], _sidecar_rec(build={**RUN, "mlx": "0.32"}))
    with pytest.raises(SystemExit, match="different build"):
        kl_pair.main()
    _arm(tmp_path, "x", [1, 2, 3, 5], None)
    (tmp_path / "x__prose.safetensors.json").unlink()
    with pytest.raises(SystemExit, match="no .json sidecar"):
        kl_pair.main()
    monkeypatch.setattr(sys, "argv", argv + ["--allow-unverified-pair"])
    assert kl_pair.main() == 0
    assert "UNVERIFIED PAIRING" in capsys.readouterr().out


def test_kl_ladder_preflight(tmp_path, monkeypatch):
    from vqlab.score import kl_ladder as KL
    c = tmp_path / "c"
    c.mkdir()
    (c / "meta.json").write_text(json.dumps({"numerics": {**RUN, "mlx": "0.32.0.dev"}}))
    monkeypatch.setattr(KL, "interpreter_build", lambda py, d: dict(RUN))
    with pytest.raises(SystemExit, match="mlx cache '0.32.0.dev'"):
        KL._preflight_builds("py", {"r": "/x"}, {}, {"prose": str(c)}, False)
    assert KL._preflight_builds("py", {"r": "/x"}, {}, {"prose": str(c)}, True) == RUN
    # old cache: warn, check only the variant it carries
    (c / "meta.json").write_text(json.dumps({"scorer_variant": "shared-clamp"}))
    assert KL._preflight_builds("py", {"r": "/x"}, {}, {"prose": str(c)}, False) == RUN


def test_card_tables_measured_with(tmp_path, capsys):
    from vqlab.records import card_tables as CT
    cell = {"mean_kl_millinats": 1.0, "numerics": RUN}
    j = tmp_path / "k.json"
    j.write_text(json.dumps({"table": {"a": {c: cell for c in CT.CORPORA}},
                             "rungs": {"a": "/nope"}, "reference": "a"}))
    CT.main(["--kl", str(j)])
    assert "Measured with: mlx 0.31.2, mlx-lm 0.31.9, variant shared-clamp." in capsys.readouterr().out
    old = {"mean_kl_millinats": 1.0}
    j.write_text(json.dumps({"table": {"a": {c: old for c in CT.CORPORA},
                                       "b": {c: cell for c in CT.CORPORA}},
                             "rungs": {"a": "/nope", "b": "/nope"}, "reference": "a"}))
    CT.main(["--kl", str(j)])
    assert "MIXED builds" in capsys.readouterr().out


def test_runtime_equiv_compare_and_main(tmp_path, monkeypatch, capsys):
    from vqlab.score import runtime_equiv as RE
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert RE.compare(a, a.copy())["bitwise"]
    z = np.zeros((1, 2), np.float32)
    assert not RE.compare(z, -z)["bitwise"]                  # -0.0 is not 0.0, bitwise
    b = a.copy()
    b[1, 3] += 0.5
    r = RE.compare(a, b)
    assert not r["bitwise"] and r["max_abs_logit_diff"] == 0.5 and r["positions_differing"] == 1
    assert RE.compare(a, a[:2])["note"].startswith("different shapes")

    def fake_side(python, model, out, tokens, chunk, knurlogic):
        np.save(out, a if python == "pa" else b)
        return {"numerics": {**RUN, "mlx": python}, "ids_head": [1], "knurlogic": knurlogic}, None
    monkeypatch.setattr(RE, "_run_side", fake_side)
    out = tmp_path / "eq.json"
    rc = RE.main(["--model", "m", "--python-a", "pa", "--python-b", "pb",
                  "--keep-dir", str(tmp_path), "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 1 and "builds differ in: mlx" in text and "DIFFERENT: max |d logit| 0.5" in text
    assert json.loads(out.read_text())["result"]["bitwise"] is False


def test_runtime_equiv_is_a_routed_command():
    import pathlib
    from vqlab import cli
    assert cli.COMMANDS["runtime-equiv"][0] == "runtime_equiv.py"
    root = pathlib.Path(__file__).resolve().parents[1]
    assert "runtime-equiv" in cli.routed((root / "CONTEXT.md").read_text())
    assert "`runtime-equiv`" in (root / "src/vqlab/score/CONTEXT.md").read_text()
