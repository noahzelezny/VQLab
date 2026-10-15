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
def fams(lab_dirs, monkeypatch):
    root, teach = lab_dirs.families, lab_dirs.teachers

    def add(family, teacher, kind, layers, vision=0.8, visual_keys=True,
            mtp=0.0, wiring=None):
        src = teach / teacher
        src.mkdir(parents=True)
        key = "model.visual.blocks.0.w" if visual_keys else "vision_tower.blocks.0.w"
        (src / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {key: "a"}}))
        d = root / family / "teachers" / teacher
        d.mkdir(parents=True)
        by = {"carry": 1.0, "embed": 1.0, "vq_target": 16.0, "vision": vision}
        if mtp:
            by["mtp"] = mtp
        (d / "profile.json").write_text(json.dumps({
            "identity": {"path": str(src)},
            "arch": {"kind": kind, "vq_layers": list(range(layers)),
                     "model_type": "qwen3_5_moe" if kind == "moe" else "qwen3_5"},
            "bytes": {"by_class_gib": by},
            "geometries": _geos([(2, 512), (4, 64), (4, 128), (4, 256), (4, 512), (8, 4096)])}))
        if wiring:
            rec = {"wiring": wiring, "source": "a fake sidecar"}
            (d / "mtp.json").write_text(json.dumps(rec))
        return d
    monkeypatch.setattr(rq.registry, "family_dirs", lambda: [root])
    monkeypatch.setattr(rq.registry, "teacher_caches", lambda t=None: [])
    # a box every fake build fits on: compiled output must not depend on the
    # RAM of the machine the tests run on
    monkeypatch.setattr(rq.preflight_ram, "ram_bytes", lambda: 1 << 60)
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


def test_expected_files_expand_where_the_step_runs(fams, tmp_path):
    fams("qwen3_5", "T", "moe", 4)
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
                              "vision": True})
    assert _names(q)[2:4] == ["vision", "vision-smoke"]
    assert _args(q, "vision")[-3:] == ["--dest-prefix", "vision_tower", "--permute-conv5"]
    fams("qwen3_5", "U", "moe", 4, visual_keys=False)
    q, _ = rq.compile_recipe({"name": "b", "family": "qwen3_5", "teacher": "U", "geo": "d4-K256",
                                  "vision": True})
    assert "--dest-prefix" not in _args(q, "vision")


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
    assert _args(q, "convert")[-5:] == ["--bits", "3", "--family", "qwen3_5", "--cpu-load"]


def test_moe_converts_bind_teacher_reads_to_the_cpu_stream(fams):
    """A cold multi-GB lazy read inside a GPU command buffer trips the Metal
    watchdog (the 397B teacher's embedding, 2026-10-13); --cpu-load is
    byte-identical on the 35B, so every MoE convert passes it."""
    fams("qwen3_5", "T", "moe", 4)
    q, _ = rq.compile_recipe({"name": "a", "family": "qwen3_5", "teacher": "T", "geo": "d4-K256"})
    assert _args(q, "struct-base")[-1] == "--cpu-load"


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
    ({"tail": "30"}, "tail is"),
    ({"tail": {"from": "x", "geo": "d4-K512"}}, "tail.from must be an integer"),
    ({"protect_bits": [8]}, "protect_bits must be an integer"),
    ({"bits": True}, "bits must be an integer"),
    ({"mtp": True}, "T has no MTP head"),
])
def test_refusals(fams, recipe, why):
    fams("qwen3_5", "T", "moe", 40)
    base = {"name": "a", "family": "qwen3_5", "teacher": "T", "geo": "d4-K256"}
    with pytest.raises(rq.RecipeError, match=why):
        rq.compile_recipe({**base, **recipe})


def test_hf_cache_teacher_stays_portable(fams, monkeypatch, tmp_path):
    d = fams("qwen3_5", "H", "moe", 4)
    snap = tmp_path / "hub" / "models--org--m" / "snapshots" / ("ab12cd3" + "0" * 33)
    snap.mkdir(parents=True)
    (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    prof = json.loads((d / "profile.json").read_text())
    prof["identity"]["path"] = "hf://org/m@ab12cd3"
    (d / "profile.json").write_text(json.dumps(prof))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    assert config.expand("--src=hf://org/m@ab12cd3") == f"--src={snap}"
    q, _ = rq.compile_recipe({"name": "a", "family": "qwen3_5", "teacher": "H", "geo": "d4-K256",
                              "vision": True})
    assert "hf://org/m@ab12cd3" in _args(q, "fit")
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


def test_malformed_inputs_are_recipe_errors(fams):
    with pytest.raises(rq.RecipeError, match="JSON object"):
        rq.compile_recipe(["not", "a", "recipe"])
    d = fams("qwen3_5", "T", "moe", 40)
    prof = json.loads((d / "profile.json").read_text())
    del prof["bytes"]
    (d / "profile.json").write_text(json.dumps(prof))
    with pytest.raises(rq.RecipeError, match="malformed"):
        rq.compile_recipe({"name": "a", "family": "qwen3_5", "teacher": "T", "geo": "d4-K256"})


def test_non_contiguous_vq_layers_are_passed_exactly(fams):
    d = fams("qwen3_5", "T", "moe", 40)
    prof = json.loads((d / "profile.json").read_text())
    prof["arch"]["vq_layers"] = [0, 1, 2, 5, 6, 7]
    (d / "profile.json").write_text(json.dumps(prof))
    q, _ = rq.compile_recipe({"name": "a", "family": "qwen3_5", "teacher": "T", "geo": "d4-K256",
                              "tail": {"from": 6, "geo": "d4-K512"}})
    fit = _args(q, "fit")
    assert fit[fit.index("--vq-layers") + 1] == "0,1,2,5,6,7"
    with pytest.raises(rq.RecipeError, match="tail.from"):         # 4 is not a VQ layer
        rq.compile_recipe({"name": "a", "family": "qwen3_5", "teacher": "T", "geo": "d4-K256",
                           "tail": {"from": 4, "geo": "d4-K512"}})


def _st(pf, **kw):
    return {"name": "s", "cmd": "fit-moe", "args": ["--out", "x"], "preflight": pf, **kw}


def test_writes_gib_rides_on_any_preflight_but_skip():
    assert steps.validate({"steps": [_st({"args": ["--small"], "writes_gib": 3})]}) == []
    [err] = steps.validate({"steps": [_st({"skip": "why", "writes_gib": 3})]})
    assert "any of them but skip" in err


def test_a_preflight_never_retries():
    assert steps.retries(_st({"skip": "t"}, retries=3), True) == 0
    assert steps.retries(_st({"skip": "t"}, retries=3), False) == 3


def test_scratch_placeholder_in_expected_files(lab_dirs, tmp_path):
    st = _st({"skip": "t"}, expect={"files": ["<scratch>/x"]})
    assert steps.expect(st, pathlib.Path("/tree"), False)["files"] == [str(tmp_path / "scratch" / "x")]


def test_preflight_writes_gib_beyond_free_space_is_refused(lab_dirs, monkeypatch, tmp_path):
    monkeypatch.setattr(steps.shutil, "disk_usage", lambda p: type("U", (), {"free": 50 << 30})())
    st = _st({"args": ["--small"], "writes_gib": 100})
    args, verdict = steps.preflight_args(st, st["args"], {"qdir": str(tmp_path / "q")}, {})
    assert args is None and verdict["verdict"] == "fail" and "not enough scratch" in verdict["reasons"][0]
    st = _st({"args": ["--small"], "writes_gib": 1})
    args, verdict = steps.preflight_args(st, st["args"], {"qdir": str(tmp_path / "q")}, {})
    assert verdict is None and args[0] == "--small"


def test_a_short_hf_rev_is_not_a_reference():
    assert config.expand("hf://org/m@ab12") == "hf://org/m@ab12"


def test_a_bad_value_in_the_profile_is_a_recipe_error(fams):
    d = fams("qwen3_5", "T", "moe", 40)
    prof = json.loads((d / "profile.json").read_text())
    prof["bytes"]["by_class_gib"]["carry"] = "lots"
    (d / "profile.json").write_text(json.dumps(prof))
    with pytest.raises(rq.RecipeError, match="malformed"):
        rq.compile_recipe({"name": "a", "family": "qwen3_5", "teacher": "T", "geo": "d4-K256"})


GOLDEN = pathlib.Path(__file__).parent / "fixtures" / "recipe_golden.json"
_GOLDEN_RECIPES = {
    "moe": ("qwen3_5", "T", {"geo": "d4-K256", "vision": True, "protect_bits": 6}),
    "moe-tail": ("qwen3_5", "T", {"geo": "d4-K256", "tail": {"from": 30, "geo": "d4-K512"}}),
    "moe-d8": ("qwen3_5", "T", {"geo": "d8-K4096", "mtp": True}),
    "moe-sz-mtp": ("qwen3_5", "T", {"geo": "d4-K256", "vision": True, "skipzero": True,
                                    "mtp": True}),
    "dense-mtp-probe": ("qwen3_8_dense", "P", {"geo": "d4-K256", "mtp": True}),
    "moe-affine": ("qwen3_5", "T", {"mode": "affine", "bits": 3, "group": 32, "vision": True}),
    "moe-affine-protect": ("qwen3_5", "T", {"mode": "affine", "bits": 2, "protect_bits": 8}),
    "moe-affine-g128": ("qwen3_5", "T", {"mode": "affine", "bits": 4, "group": 128}),
    "dense": ("qwen3_8_dense", "D", {"geo": "d2-K512", "bits": 4, "vision": True}),
    "dense-score": ("qwen3_8_dense", "D", {"geo": "d4-K256", "score": True}),
    "moe-sz": ("qwen3_5", "T", {"geo": "d4-K256", "vision": True, "skipzero": True}),
    "moe-sz-tail": ("qwen3_5", "T", {"geo": "d4-K256", "tail": {"from": 30, "geo": "d8-K4096"},
                                     "skipzero": True, "score": True}),
}


def _measured_dead_rows(d, rows=(("gate_proj", 1024), ("up_proj", 1024), ("down_proj", 4096))):
    """A teacher folder `d` with dead rows measured (skipzero.json) and each
    geometry's per-module bytes in its profile: 1 MiB per 1 bit per weight."""
    prof = json.loads((d / "profile.json").read_text())
    for es in prof["geometries"].values():
        for e in es:
            e["bytes"] = (1 << 20) * (e["K"].bit_length() - 1) // e["d"]
    (d / "profile.json").write_text(json.dumps(prof))
    (d / "skipzero.json").write_text(json.dumps({
        "source": "a fake build", "rows": dict(rows),
        "dead_rows": {"0": {"gate_proj": 512, "up_proj": 256, "down_proj": 0},
                      "35": {"down_proj": 1024}}}))


# the 397B's recorded wiring (families/qwen3_5/teachers/<397B>/mtp.json)
_WIRING = {"fc_order": "eh", "h_source": "pre_norm"}


def _golden_outputs(fams, tmp_path, monkeypatch):
    _measured_dead_rows(fams("qwen3_5", "T", "moe", 40, mtp=1.5, wiring=_WIRING))
    fams("qwen3_8_dense", "D", "dense", 24, vision=0)
    fams("qwen3_8_dense", "P", "dense", 24, vision=0, mtp=0.25)
    monkeypatch.setattr(rq.registry, "teacher_caches", lambda t=None: [
        {"teacher": "D", "corpus": "prose", "path": "/c/prose"}])
    out = {}
    for key, (family, teacher, extra) in _GOLDEN_RECIPES.items():
        q, warns = rq.compile_recipe({"name": key, "family": family, "teacher": teacher, **extra})
        out[key] = json.loads(json.dumps([q, warns]).replace(str(tmp_path), "<tmp>"))
    return out


def test_compiled_queues_are_unchanged(fams, tmp_path, monkeypatch):
    """A refactor of the compiler must not move one byte of its output."""
    # compared as text, so a reordered key is a change too
    assert json.dumps(_golden_outputs(fams, tmp_path, monkeypatch)) == json.dumps(json.loads(GOLDEN.read_text()))


def test_skipzero_packs_the_finished_artifact_and_checks_run_on_the_copy(fams):
    _measured_dead_rows(fams("qwen3_5", "T", "moe", 40))
    base = {"name": "a", "family": "qwen3_5", "teacher": "T", "geo": "d4-K256", "vision": True}
    plain, _ = rq.compile_recipe(base)
    q, warns = rq.compile_recipe({**base, "skipzero": True})
    assert _names(q) == ["struct-base", "fit", "vision", "vision-smoke", "skipzero",
                         "skipzero-check", "skipzero-bundle", "smoke", "runtime-check",
                         "release-check"]
    built = _args(q, "fit")[_args(q, "fit").index("--out") + 1]
    pack = next(s for s in q["steps"] if s["name"] == "skipzero")
    sz = pack["args"][pack["args"].index("--out") + 1]
    assert pack["cmd"] == "sz-pack" and pack["args"][0] == built and sz.startswith("<scratch>/")
    assert pack["preflight"] == {"append": ["--layers", "0-0"]} and pack["gpu"] is False
    check = next(s for s in q["steps"] if s["name"] == "skipzero-check")
    assert check["args"] == [sz, built] and check["gpu"] is False
    # as the published 397B builds: sz-pack -> bundle the packed copy -> pin
    bundle = next(s for s in q["steps"] if s["name"] == "skipzero-bundle")
    assert bundle["cmd"] == "bundle" and bundle["args"] == ["--artifact", sz]
    assert "--runtime" not in bundle["args"]      # keeps the profile the fit baked in
    assert check["preflight"]["writes_gib"] == 0
    assert _args(q, "vision-smoke")[0] == built
    assert _args(q, "smoke")[0] == sz and _args(q, "runtime-check")[1] == sz
    assert _args(q, "release-check")[1] == sz
    assert steps.validate(q) == [] and any("EXPERIMENTAL" in w for w in warns)
    # d4-K256 = 2 bits per weight -> 2 MiB a module; minus a 1-bit-per-row mask per packed module
    saved = (512 / 1024 + 256 / 1024 + 1024 / 4096) * (2 << 20) - (1024 + 1024 + 4096) / 8
    assert q["estimate"]["artifact_gib"] == round(plain["estimate"]["artifact_gib"] - saved / 2 ** 30, 2)


def test_skipzero_prices_each_layer_at_its_own_geometry(fams):
    _measured_dead_rows(fams("qwen3_5", "T", "moe", 40))
    prof = rq.load_profile("qwen3_5", "T")
    sz = rq.registry.teacher_record("qwen3_5", "T", "skipzero.json")
    flat = rq.skipzero_gib(prof, sz, {i: (4, 256) for i in range(40)})
    mixed = rq.skipzero_gib(prof, sz, {i: (4, 256) if i < 30 else (8, 4096) for i in range(40)})
    # layer 35's dead rows are priced at d8-K4096 (1.5 bits) instead of d4-K256 (2 bits)
    assert mixed == pytest.approx(flat - 1024 / 4096 * (0.5 * (1 << 20)) / 2 ** 30)


@pytest.mark.parametrize("recipe, why", [
    ({"mode": "affine"}, "only on MoE VQ"),
    ({"skipzero": "yes"}, "true or false"),
])
def test_skipzero_refusals(fams, recipe, why):
    _measured_dead_rows(fams("qwen3_5", "T", "moe", 40))
    with pytest.raises(rq.RecipeError, match=why):
        rq.compile_recipe({"name": "a", "family": "qwen3_5", "teacher": "T", "geo": "d4-K256",
                           "skipzero": True, **recipe})


def test_skipzero_needs_measured_dead_rows(fams):
    fams("qwen3_5", "T", "moe", 40)
    fams("qwen3_8_dense", "D", "dense", 24, vision=0)
    with pytest.raises(rq.RecipeError, match="no dead rows are measured"):
        rq.compile_recipe({"name": "a", "family": "qwen3_5", "teacher": "T", "geo": "d4-K256",
                           "skipzero": True})
    with pytest.raises(rq.RecipeError, match="only on MoE VQ"):
        rq.compile_recipe({"name": "a", "family": "qwen3_8_dense", "teacher": "D", "geo": "d4-K256",
                           "skipzero": True})


_MOE = {"name": "a", "family": "qwen3_5", "teacher": "T", "geo": "d4-K256"}


def _step_of(q, name):
    return next(s for s in q["steps"] if s["name"] == name)


def test_mtp_packs_the_recorded_wiring_into_the_finished_artifact(fams):
    """As every published 397B sidecar was made: the bf16 head extracted
    from the teacher, packed at mtp-pack's default recipe with the wiring
    the shipped head records, into the copy that ships, then gated."""
    _measured_dead_rows(fams("qwen3_5", "T", "moe", 40, mtp=1.5, wiring=_WIRING))
    plain, _ = rq.compile_recipe({**_MOE, "skipzero": True})
    q, _ = rq.compile_recipe({**_MOE, "skipzero": True, "mtp": True})
    assert _names(q) == ["struct-base", "fit", "skipzero", "skipzero-check",
                         "skipzero-bundle", "mtp-extract", "mtp-pack", "mtp-smoke",
                         "mtp-accept", "smoke", "runtime-check", "release-check"]
    sz = _args(q, "smoke")[0]
    side = f"{sz}/mtp-head-q6.safetensors"
    graft = _args(q, "mtp-extract")[_args(q, "mtp-extract").index("--out") + 1]
    assert _args(q, "mtp-pack") == ["--model", sz, "--mtp", graft, "--fc-order", "eh",
                                    "--h-source", "pre_norm", "--out", side]
    assert _args(q, "mtp-smoke") == ["--model", sz, "--sidecar", side]
    acc = _args(q, "mtp-accept")
    assert f"head={side}" in acc
    assert acc[acc.index("--min-acceptance") + 1] == str(rq.MTP_ACCEPT_FLOOR)
    assert steps.validate(q) == []
    # the sidecar: the profile's bf16 head at 6 bits + a 16-bit scale and bias per 32
    want = plain["estimate"]["artifact_gib"] + 1.5 * 7 / 16
    assert q["estimate"]["artifact_gib"] == round(want, 2)


def test_mtp_without_a_recorded_wiring_is_probed_first(fams):
    fams("qwen3_8_dense", "P", "dense", 24, vision=0, mtp=0.25)
    q, warns = rq.compile_recipe({"name": "p", "family": "qwen3_8_dense",
                                  "teacher": "P", "geo": "d4-K256", "mtp": True})
    i = _names(q).index("mtp-extract")
    assert _names(q)[i:i + 6] == ["mtp-extract", "mtp-trial-pack", "mtp-wiring-probe",
                                  "mtp-pack", "mtp-smoke", "mtp-accept"]
    probe = _step_of(q, "mtp-wiring-probe")
    pa = probe["args"]
    trial = _args(q, "mtp-trial-pack")
    assert probe["cmd"] == "mtp-probe35" and "--fc-order" not in trial
    assert pa[pa.index("--sidecar") + 1] == trial[-1]
    pack = _args(q, "mtp-pack")
    assert pack[pack.index("--wiring-from") + 1] == pa[pa.index("--out") + 1]
    assert "--fc-order" not in pack
    assert any("probe chooses it" in w for w in warns) and steps.validate(q) == []


def test_mtp_unrecorded_wiring_too_large_to_probe_is_refused(fams, monkeypatch):
    fams("qwen3_5", "T", "moe", 40, mtp=1.5)
    _small_box(monkeypatch)
    with pytest.raises(rq.RecipeError, match="no MTP wiring is recorded"):
        rq.compile_recipe({**_MOE, "mtp": True})


def test_mtp_oversized_serves_the_sidecar_from_the_linked_folder(fams, monkeypatch):
    fams("qwen3_5", "T", "moe", 40, mtp=1.5, wiring=_WIRING)
    _small_box(monkeypatch)
    q, warns = rq.compile_recipe({**_MOE, "mtp": True})
    names = _names(q)
    assert "mtp-accept" not in names
    assert names.index("mtp-pack") < names.index("knurlogic-link")
    out = _args(q, "knurlogic-link")[0]
    assert _args(q, "mtp-pack")[-1] == f"{out}/mtp-head-q6.safetensors"
    assert any("gated by check-release" in w for w in warns)


@pytest.mark.parametrize("arch, record, why", [
    ({"model_type": "gemma4"}, None, "packs no MTP head for model_type 'gemma4'"),
    ({}, {"fc_order": "eh"}, "must set exactly fc_order, h_source"),
])
def test_mtp_refusals(fams, arch, record, why):
    d = fams("qwen3_5", "T", "moe", 4, mtp=1.5, wiring=record)
    prof = json.loads((d / "profile.json").read_text())
    prof["arch"].update(arch)
    (d / "profile.json").write_text(json.dumps(prof))
    with pytest.raises(rq.RecipeError, match=why):
        rq.compile_recipe({**_MOE, "mtp": True})


def test_the_shipped_heads_record_their_wiring():
    """The checked-in records name exactly the choices their family leaves open."""
    from vqlab.mtp import registry as mtp_registry
    for prof_file in config.checkout_families_dir().glob("*/teachers/*/mtp.json"):
        rec = json.loads(prof_file.read_text())
        prof = json.loads((prof_file.parent / "profile.json").read_text())
        spec = mtp_registry.FAMILIES[prof["arch"]["model_type"]]
        assert sorted(rec["wiring"]) == sorted(spec.wiring) and rec["source"]


def test_skipzero_dims_are_sz_packs():
    from vqlab.skipzero import skipzero_load
    assert rq.SZ_DIMS == skipzero_load.SUPPORTED_DIMS


def test_affine_protect_and_group_are_validated():
    base = {"name": "a", "teacher": "T", "mode": "affine", "bits": 2}
    with pytest.raises(rq.RecipeError, match="no experts"):
        rq.Recipe.from_json({**base, "family": "qwen3_8_dense", "protect_bits": 8})
    with pytest.raises(rq.RecipeError, match="2-8"):
        rq.Recipe.from_json({**base, "family": "qwen3_5", "protect_bits": 16})
    with pytest.raises(rq.RecipeError, match="32, 64 or 128"):
        rq.Recipe.from_json({**base, "family": "qwen3_5", "group": 48})
    assert rq.Recipe.from_json({**base, "family": "qwen3_5"}).protect_bits is None


GOLDEN_OVERSIZED = pathlib.Path(__file__).parent / "fixtures" / "recipe_golden_oversized.json"
_OVERSIZED_RECIPES = {
    "moe-sz-big": ("qwen3_5", "T", {"geo": "d4-K256", "tail": {"from": 30, "geo": "d8-K4096"},
                                    "vision": True, "skipzero": True, "score": True}),
    "moe-big": ("qwen3_5", "T", {"geo": "d4-K256", "vision": True}),
    "dense-big": ("qwen3_8_dense", "D", {"geo": "d4-K256"}),
    "moe-sz-mtp-big": ("qwen3_5", "T", {"geo": "d4-K256", "vision": True,
                                        "skipzero": True, "mtp": True}),
}


def _small_box(monkeypatch, gib=0.5):
    """A box whose resident bar (0.90 x RAM) is `gib`."""
    monkeypatch.setattr(rq.preflight_ram, "ram_bytes", lambda: gib / 0.90 * 2 ** 30)


def _oversized_outputs(fams, tmp_path, monkeypatch):
    _measured_dead_rows(fams("qwen3_5", "T", "moe", 40, mtp=1.5, wiring=_WIRING))
    fams("qwen3_8_dense", "D", "dense", 24, vision=0)
    monkeypatch.setattr(rq.registry, "teacher_caches", lambda t=None: [
        {"teacher": "T", "corpus": "prose", "path": "/c/prose"}])
    _small_box(monkeypatch)
    out = {}
    for key, (family, teacher, extra) in _OVERSIZED_RECIPES.items():
        q, warns = rq.compile_recipe({"name": key, "family": family, "teacher": teacher, **extra})
        out[key] = json.loads(json.dumps([q, warns]).replace(str(tmp_path), "<tmp>"))
    return out


def test_oversized_queues_are_unchanged(fams, tmp_path, monkeypatch):
    assert json.dumps(_oversized_outputs(fams, tmp_path, monkeypatch)) == \
        json.dumps(json.loads(GOLDEN_OVERSIZED.read_text()))


def test_an_oversized_build_is_smoked_on_the_cluster_through_knurlogic(fams, monkeypatch):
    _measured_dead_rows(fams("qwen3_5", "T", "moe", 40))
    base = {"name": "a", "family": "qwen3_5", "teacher": "T", "geo": "d4-K256",
            "vision": True, "skipzero": True}
    small, _ = rq.compile_recipe(base)
    _small_box(monkeypatch, gib=small["estimate"]["artifact_gib"] - 0.01)
    q, warns = rq.compile_recipe(base)
    assert _names(q) == ["struct-base", "fit", "vision", "skipzero", "skipzero-check",
                         "skipzero-bundle", "smoke", "runtime-check", "knurlogic-link",
                         "release-check"]
    sz = _args(q, "smoke")[0]
    link = next(s for s in q["steps"] if s["name"] == "knurlogic-link")
    assert link["cmd"] == "knurlogic-link" and link["args"] == [sz] and link["gpu"] is False
    rel = next(s for s in q["steps"] if s["name"] == "release-check")
    assert rel["args"][:2] == ["--artifact", sz]
    assert rel["args"][-2:] == ["--knurlogic-smoke", "fit"]
    # the preflight really runs it: no skip, so a cluster that is down fails it
    assert "full" in rel["preflight"] and "skip" not in rel["preflight"]
    assert steps.validate(q) == []
    assert any("through Knurlogic" in w for w in warns)
    # everything but the smoke steps is the same queue
    keep = [s for s in small["steps"] if s["name"] not in ("vision-smoke", "release-check")]
    assert [s for s in q["steps"] if s["name"] not in ("knurlogic-link", "release-check")] == keep


def test_the_bar_is_the_estimate_against_this_boxs_ram(fams, monkeypatch):
    fams("qwen3_5", "T", "moe", 40)
    r = {"name": "a", "family": "qwen3_5", "teacher": "T", "geo": "d4-K256"}
    est = rq.compile_recipe(r)[0]["estimate"]["artifact_gib"]
    _small_box(monkeypatch, gib=est + 0.01)     # just fits
    assert "knurlogic-link" not in _names(rq.compile_recipe(r)[0])
    _small_box(monkeypatch, gib=est - 0.01)     # just does not
    assert "knurlogic-link" in _names(rq.compile_recipe(r)[0])


def test_the_bar_is_read_from_the_machine(monkeypatch):
    calls = []
    pages = {"SC_PHYS_PAGES": 96 << 18, "SC_PAGE_SIZE": 4096}
    monkeypatch.setattr(rq.preflight_ram.os, "sysconf", lambda k: calls.append(k) or pages[k])
    assert rq.preflight_ram.bar_gib() == pytest.approx(96 * 0.90)
    assert sorted(calls) == ["SC_PAGE_SIZE", "SC_PHYS_PAGES"]


def test_preflight_check_compares_shard_bytes_with_the_bar(tmp_path, monkeypatch):
    (tmp_path / "a.safetensors").write_bytes(b"\0" * 900)
    monkeypatch.setattr(rq.preflight_ram, "ram_bytes", lambda: 1000)
    assert rq.preflight_ram.check(tmp_path) == (True, 900, 1000)
    ok, text = rq.preflight_ram.report(tmp_path, headroom=0.5)
    assert not ok and "PREFLIGHT FAIL" in text
    assert rq.preflight_ram.main([str(tmp_path), "--headroom", "0.5"]) == 2


GUI = pathlib.Path(__file__).resolve().parents[1] / "src/vqlab/agents/gui_static"
# recipe JSON -> the Recipe page's draft for the same build
_PARITY = [
    ("qwen3_5", {"mode": "affine", "bits": 4, "group": 64},
     {"mode": "affine", "bits": 4, "group": 64, "aprotect": 0}),
    ("qwen3_5", {"mode": "affine", "bits": 3, "group": 32, "protect_bits": 6, "vision": True},
     {"mode": "affine", "bits": 3, "group": 32, "aprotect": 6, "tower": True}),
    ("qwen3_5", {"geo": "d4-K256", "protect_bits": 6, "vision": True},
     {"mode": "vq", "geo": "d4-K256", "protect": 6, "tower": True}),
    ("qwen3_5", {"geo": "d4-K256", "tail": {"from": 20, "geo": "d8-K4096"}},
     {"mode": "vq", "geo": "d4-K256", "protect": 8, "tailOn": True, "tailGeo": "d8-K4096",
      "tailFrom": 20}),
    ("qwen3_5", {"geo": "d4-K256", "skipzero": True},
     {"mode": "vq", "geo": "d4-K256", "protect": 8, "skipzero": True}),
    ("qwen3_5", {"geo": "d4-K256", "skipzero": True, "tail": {"from": 30, "geo": "d8-K4096"}},
     {"mode": "vq", "geo": "d4-K256", "protect": 8, "skipzero": True, "tailOn": True,
      "tailGeo": "d8-K4096", "tailFrom": 30}),
    ("qwen3_8_dense", {"geo": "d4-K256", "bits": 6, "group": 32},
     {"mode": "vq", "geo": "d4-K256", "bits": 6, "group": 32}),
    ("qwen3_5", {"geo": "d4-K256", "skipzero": True, "mtp": True},
     {"mode": "vq", "geo": "d4-K256", "protect": 8, "skipzero": True, "mtp": True}),
    ("qwen3_8_dense", {"geo": "d4-K256", "bits": 6, "group": 32, "mtp": True},
     {"mode": "vq", "geo": "d4-K256", "bits": 6, "group": 32, "mtp": True}),
]


def _page_size_js(page: pathlib.Path) -> str:
    """The Recipe page's size arithmetic, verbatim: isMoe .. sizeRaw."""
    src = page.read_text()
    return src[src.index("const isMoe=t=>"):src.index("function sanitize(")]


def test_the_recipe_page_sizes_a_build_as_the_compiler_does(fams):
    """The live drag re-derives the compiler's artifact size in JS (instant
    feedback before the preview returns); run that JS against
    compile_recipe's estimate so the two cannot drift."""
    import shutil
    import subprocess
    from vqlab.agents import teachers as T
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    js = _page_size_js(GUI / "index.html")
    assert js == _page_size_js(GUI / "bench" / "bench.html")
    for d in (fams("qwen3_5", "T", "moe", 40, mtp=1.5, wiring=_WIRING),
              fams("qwen3_8_dense", "D", "dense", 24, vision=0, mtp=0.25)):
        _measured_dead_rows(d)
        prof = json.loads((d / "profile.json").read_text())
        for es in prof["geometries"].values():       # GiB-scale stacks and dead rows, so
            for e in es:                             # every case prices a different size
                bits = (e["K"].bit_length() - 1) / e["d"]
                e["bytes"] *= 64
                e["stack_gib"] = 16 / 3 * bits / 16
                e["bits_per_weight"] = bits
        (d / "profile.json").write_text(json.dumps(prof))
    cases, want = [], []
    for family, recipe, draft in _PARITY:
        teacher = "T" if family == "qwen3_5" else "D"
        prof = rq.load_profile(family, teacher)
        vq = prof["arch"]["vq_layers"]
        t = {"kind": prof["arch"]["kind"], "gib": prof["bytes"]["by_class_gib"],
             "geometries": T._geometries(prof), "vq_layers": len(vq), "first_vq": min(vq),
             "skipzero": T._skipzero(family, teacher, prof),
             "mtp_gib": T._mtp_gib(family, teacher, prof)}
        cases.append([t, draft])
        q, _ = rq.compile_recipe({"name": "a", "family": family, "teacher": teacher, **recipe})
        want.append(q["estimate"]["artifact_gib"])
    run = js + "\nconst c=JSON.parse(require('fs').readFileSync(0,'utf8'));" \
               "console.log(JSON.stringify(c.map(([t,r])=>sizeOf(t,r).gib)));"
    out = subprocess.run([node, "-e", run], input=json.dumps(cases), capture_output=True,
                         text=True, check=True).stdout
    got = json.loads(out)
    assert [round(g, 2) for g in got] == want, (got, want)
    assert len(set(want)) == len(want)              # every case prices a different build
