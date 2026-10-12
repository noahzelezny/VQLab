"""plan/recipe_queue.py: a recipe compiles to the build chain that has run,
and to nothing else."""
import json
import pathlib

import pytest

from vqlab import config
from vqlab.plan import recipe_queue as rq
from vqlab.queue import steps


def _geos(rows):
    return {p: [{"d": d, "K": k, "legal": True} for d, k in rows]
            for p in ("gate_proj", "up_proj", "down_proj")}


@pytest.fixture
def fams(tmp_path, monkeypatch):
    root = tmp_path / "families"
    teach = tmp_path / "teachers"

    def add(family, teacher, kind, layers, vision=0.8, visual_keys=True):
        src = teach / teacher
        src.mkdir(parents=True)
        key = "model.visual.blocks.0.w" if visual_keys else "vision_tower.blocks.0.w"
        (src / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {key: "a"}}))
        d = root / family / "teachers" / teacher
        d.mkdir(parents=True)
        (d / "profile.json").write_text(json.dumps({
            "identity": {"path": str(src)}, "arch": {"kind": kind, "vq_layers": list(range(layers))},
            "bytes": {"by_class_gib": {"carry": 1.0, "embed": 1.0, "vq_target": 16.0, "vision": vision}},
            "geometries": _geos([(2, 512), (4, 64), (4, 128), (4, 256), (4, 512), (8, 4096)])}))
        return d
    monkeypatch.setattr(rq.registry, "_family_dirs", lambda: [root])
    monkeypatch.setattr(rq.registry, "teacher_caches", lambda t=None: [])
    return add


def _names(q):
    return [s["name"] for s in q["steps"]]


def _args(q, name):
    return next(s for s in q["steps"] if s["name"] == name)["args"]


def test_moe_flat(fams):
    fams("qwen3_5", "T", "moe", 40)
    q, warns = rq.compile_recipe({"name": "a", "family": "qwen3_5", "teacher": "T", "geo": "d4-K256"})
    assert _names(q) == ["struct-base", "fit", "smoke", "runtime-check", "release-check"]
    fit = _args(q, "fit")
    assert fit[fit.index("--vq-layers") + 1] == "0-39" and "--pack" in fit
    assert fit[fit.index("--relerr-abort") + 1] == "0.45"
    assert _args(q, "smoke")[0] == fit[fit.index("--out") + 1]
    assert steps.validate(q) == [] and warns == []
    pre = next(s for s in q["steps"] if s["name"] == "fit")["preflight"]
    # carry + embed at protect 8 (8.5 bpw) + experts as 2-bit markers (2.5 bpw)
    assert pre["writes_gib"] == round(2.0 * 8.5 / 16 + 16.0 * 2.5 / 16, 1)
    assert "--protect-bits" in _args(q, "struct-base")
    assert q["estimate"]["artifact_gib"] > 0


def test_expected_files_expand_where_the_step_runs(fams, monkeypatch, tmp_path):
    fams("qwen3_5", "T", "moe", 4)
    monkeypatch.setenv("VQLAB_SCRATCH", str(tmp_path / "scratch"))
    q, _ = rq.compile_recipe({"name": "a", "family": "qwen3_5", "teacher": "T", "geo": "d4-K256"})
    fit = next(s for s in q["steps"] if s["name"] == "fit")
    assert steps.expect(fit, pathlib.Path("/tree"), False)["files"] == [
        str(tmp_path / "scratch" / "a" / "a" / "config.json")]
    assert steps.retries(fit, preflight=True) == 0 and steps.retries(fit, preflight=False) == 3


def test_moe_mixed_uses_the_fitters_tail(fams):
    fams("qwen3_5", "T", "moe", 40)
    q, warns = rq.compile_recipe({"name": "a", "family": "qwen3_5", "teacher": "T", "geo": "d4-K128",
                                  "tail": {"from": 30, "geo": "d4-K512"}})
    assert any("one --relerr-abort" in w for w in warns)
    fit = _args(q, "fit")
    assert fit[fit.index("--tail-from") + 1] == "30" and fit[fit.index("--tail-geom") + 1] == "d4k512"
    assert fit[fit.index("--relerr-abort") + 1] == "0.6"         # the looser of the two
    pre = next(s for s in q["steps"] if s["name"] == "fit")["preflight"]["args"]
    assert pre[pre.index("--vq-layers") + 1] == "0,30"           # body AND tail run small
    assert all(isinstance(a, str) for a in pre)


def test_vision_graft_matches_the_teachers_layout(fams):
    fams("qwen3_5", "T", "moe", 4)
    q, _ = rq.compile_recipe({"name": "a", "family": "qwen3_5", "teacher": "T", "geo": "d4-K256",
                              "vision": True, "mtp": True})
    assert _names(q)[2:4] == ["vision", "vision-smoke"]
    assert _args(q, "vision")[-3:] == ["--dest-prefix", "vision_tower", "--permute-conv5"]
    fams("qwen3_5", "U", "moe", 4, visual_keys=False)
    q, warns = rq.compile_recipe({"name": "b", "family": "qwen3_5", "teacher": "U", "geo": "d4-K256",
                                  "vision": True, "mtp": True})
    assert "--dest-prefix" not in _args(q, "vision")
    assert any("MTP" in w for w in warns)


def test_dense_chain(fams):
    fams("qwen3_8_dense", "D", "dense", 24, vision=0)
    q, warns = rq.compile_recipe({"name": "d", "family": "qwen3_8_dense", "teacher": "D",
                                  "geo": "d2-K512", "bits": 4, "vision": True})
    assert _names(q) == ["base", "fit", "remap-check", "build", "pack", "smoke",
                         "runtime-check", "release-check"]
    assert _args(q, "build")[_args(q, "build").index("--mlp") + 1] == _args(q, "fit")[3]
    assert _args(q, "remap-check")[-1] == "--dry-run"
    assert any("no vision tower" in w for w in warns) and steps.validate(q) == []


def test_dense_with_an_incumbent_base(fams, tmp_path):
    fams("qwen3_8_dense", "D", "dense", 24, vision=0)
    inc = tmp_path / "inc"
    inc.mkdir()
    q, warns = rq.compile_recipe({"name": "d", "family": "qwen3_8_dense", "teacher": "D",
                                  "geo": "d2-K512", "base": str(inc)})
    assert _names(q)[0] == "fit" and _args(q, "build")[3] == str(inc)
    assert not any("stream-convert" in w for w in warns)


def test_affine(fams):
    fams("qwen3_5", "T", "moe", 4)
    q, _ = rq.compile_recipe({"name": "f", "family": "qwen3_5", "teacher": "T", "mode": "affine", "bits": 3})
    assert _names(q) == ["convert", "smoke", "runtime-check", "release-check"]
    assert _args(q, "convert")[-4:] == ["--bits", "3", "--family", "qwen3_5"]


def test_score_only_with_registered_caches(fams, monkeypatch):
    fams("qwen3_5", "T", "moe", 4)
    r = {"name": "a", "family": "qwen3_5", "teacher": "T", "geo": "d4-K256", "score": True}
    q, warns = rq.compile_recipe(r)
    assert "score" not in _names(q) and any("no KL caches" in w for w in warns)
    monkeypatch.setattr(rq.registry, "teacher_caches", lambda t=None: [
        {"teacher": "T", "corpus": c, "path": f"/c/{c}"} for c in ("prose", "code")])
    q, _ = rq.compile_recipe(r)
    q, warns = rq.compile_recipe(r)
    kl = _args(q, "score")
    assert kl[:4] == ["--cache", "code=/c/code", "--cache", "prose=/c/prose"]
    assert "--lazy-over-gb" in kl and any("PAIRED" in w for w in warns)


@pytest.mark.parametrize("recipe, why", [
    ({"family": "glm5_next"}, "no build of family"),
    ({"geo": "d4-K999"}, "not a legal geometry"),
    ({"geo": "d4-K64"}, "no measured healthy relerr"),
    ({"geo": "d4K256x"}, "not dD-KK"),
    ({"name": "Bad Name"}, "name:"),
    ({"teacher": "missing"}, "no profile"),
    ({"tail": {"from": 0, "geo": "d4-K512"}}, "tail.from"),
    ({"family": "qwen3_8_dense"}, "no profile"),
])
def test_refusals(fams, recipe, why):
    fams("qwen3_5", "T", "moe", 40)
    base = {"name": "a", "family": "qwen3_5", "teacher": "T", "geo": "d4-K256"}
    with pytest.raises(rq.RecipeError, match=why):
        rq.compile_recipe({**base, **recipe})


def test_hf_cache_teacher_stays_portable(fams, monkeypatch, tmp_path):
    d = fams("qwen3_5", "H", "moe", 4)
    snap = tmp_path / "hub" / "models--org--m" / "snapshots" / ("ab12" + "0" * 36)
    snap.mkdir(parents=True)
    (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    prof = json.loads((d / "profile.json").read_text())
    prof["identity"]["path"] = "hf://org/m@ab12"
    (d / "profile.json").write_text(json.dumps(prof))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    assert config.expand("--src=hf://org/m@ab12") == f"--src={snap}"
    q, _ = rq.compile_recipe({"name": "a", "family": "qwen3_5", "teacher": "H", "geo": "d4-K256",
                              "vision": True})
    assert "hf://org/m@ab12" in _args(q, "fit")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "elsewhere"))
    with pytest.raises(rq.RecipeError, match="not on this machine"):
        rq.compile_recipe({"name": "a", "family": "qwen3_5", "teacher": "H", "geo": "d4-K256"})


def test_d8_takes_its_own_threshold(fams):
    fams("qwen3_5", "T", "moe", 4)
    q, _ = rq.compile_recipe({"name": "a", "family": "qwen3_5", "teacher": "T", "geo": "d8-K4096"})
    fit = _args(q, "fit")
    assert fit[fit.index("--relerr-abort") + 1] == "0.45"


def test_kind_must_match_family(fams):
    fams("qwen3_5", "D", "dense", 4)
    with pytest.raises(rq.RecipeError, match="is dense"):
        rq.compile_recipe({"name": "a", "family": "qwen3_5", "teacher": "D", "geo": "d4-K256"})
