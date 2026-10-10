"""vqlab.api -- the stable library / MCP surface.

No GPU, no models, no network: storage roots, the runs dir, the GPU lease
and the findings log are redirected to tmp_path, and the launcher's process
spawn is replaced by a fake.
"""
from __future__ import annotations

import dataclasses
import inspect
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from vqlab import api
from vqlab.agents.mcp import tools_findings, tools_runs

ROOT = Path(__file__).resolve().parents[1]


def _js(x):
    """Every return must survive json.dumps(asdict(x)) / json.dumps(x)."""
    return json.loads(json.dumps(api.to_json(x)))


def _st(path: Path, names):
    """A header-only safetensors file holding `names` (1 byte each)."""
    hdr = {n: {"dtype": "U8", "shape": [1], "data_offsets": [i, i + 1]}
           for i, n in enumerate(names)}
    raw = json.dumps(hdr).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * len(names))


@pytest.fixture
def lab(tmp_path, monkeypatch):
    root = tmp_path / "scratch"
    art = root / "Flash-2.1-v2"
    art.mkdir(parents=True)
    (art / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    _st(art / "model-00001-of-00001.safetensors",
        ["model.layers.0.mlp.weight", "model.embed_tokens.weight"])
    (art / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": 2},
         "weight_map": {"model.layers.0.mlp.weight": "model-00001-of-00001.safetensors",
                        "model.embed_tokens.weight": "model-00001-of-00001.safetensors"}}))
    monkeypatch.setenv("VQLAB_ROOTS", str(root))
    monkeypatch.setenv("VQLAB_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("VQLAB_GPU_LEASE", str(tmp_path / "gpu.lease"))
    monkeypatch.setenv("VQLAB_LAB_STATE", str(tmp_path / "lab-state.json"))
    monkeypatch.setenv("VQLAB_HOSTNAME", "box-a")
    log = tmp_path / "FINDINGS-LOG.md"
    log.write_text("# log\n\n## F97 (2026-09-14) — packing is exact\n\nbody\n")
    monkeypatch.setattr(tools_findings, "FINDINGS_LOG", log)
    monkeypatch.setattr(tools_findings, "_FLOCK_DIR", tmp_path / ".f-locks")
    monkeypatch.setattr(tools_runs, "_exo_instances", lambda: [])
    return {"root": root, "art": art, "tmp": tmp_path}


class _FakeProc:
    pid = 4242


@pytest.fixture
def launched(lab, monkeypatch):
    """Patch the launcher's spawn: record argv, start nothing."""
    calls = []

    def fake_popen(argv, **kw):
        calls.append(argv)
        return _FakeProc()
    monkeypatch.setattr(tools_runs.subprocess, "Popen", fake_popen)
    return calls


# --------------------------------------------------------------------------- #
# The surface itself
# --------------------------------------------------------------------------- #

def test_import_loads_no_mlx():
    code = ("import sys; import vqlab.api; "
            "bad = [m for m in sys.modules if m.split('.')[0] in ('mlx', 'mlx_lm', 'numpy')]; "
            "print(bad)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]"


def test_publish_is_not_exposed():
    assert not hasattr(api, "publish")
    assert not any("publish" == n or n.startswith("publish") for n in api.__all__)
    assert "publish" not in {c.name for c in api.commands() if c.mcp}


def test_every_public_function_is_keyword_only_and_documented():
    for name in api.__all__:
        obj = getattr(api, name)
        if inspect.isclass(obj) or name == "to_json":
            continue
        sig = inspect.signature(obj)
        assert all(p.kind is p.KEYWORD_ONLY for p in sig.parameters.values()), name
        assert obj.__doc__ and len(obj.__doc__) > 40, name
        assert "return" in obj.__annotations__, name


def test_commands():
    cs = api.commands()
    assert [c.name for c in cs] == sorted(c.name for c in cs)
    by = {c.name: c for c in cs}
    assert by["fit-moe"].mcp and by["fit-moe"].gpu and by["fit-moe"].stage == "fit"
    assert _js(cs)[0]["name"] == cs[0].name


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #

def test_run_returns_a_handle_from_the_mcp_launcher(launched, lab):
    h = api.run(command="kl-ladder", args=["--cache", str(lab["root"])], tag="t1")
    assert isinstance(h, api.RunHandle) and h.pid == 4242
    assert h.log_path == str(Path(h.run_dir) / "job.log")
    assert "kl-ladder" in h.run_id and h.command.startswith("vqlab kl-ladder")
    assert launched[0][-2:] == ["--supervise", h.run_dir]
    meta = json.loads((Path(h.run_dir) / "meta.json").read_text())
    assert meta["args"] == ["--cache", str(lab["root"])]
    assert _js(h)["run_id"] == h.run_id


def test_run_refusals(launched, lab, monkeypatch):
    with pytest.raises(api.ApiError) as e:
        api.run(command="publish")
    assert e.value.code == "NOT_ALLOWED" and "publish" not in e.value.extra["allowlist"]
    with pytest.raises(api.ApiError) as e:
        api.run(command="score", args=["/definitely/not/a/root"])
    assert e.value.code == "OUTSIDE_ROOTS"
    with pytest.raises(api.ApiError) as e:
        api.run(command="score", args="--x")
    assert e.value.code == "INVALID_ARG"
    with pytest.raises(api.ApiError):
        api.run(command="")
    monkeypatch.setattr(tools_runs, "_exo_instances", lambda: ["inst-1"])
    with pytest.raises(api.ApiError) as e:
        api.run(command="score")
    assert e.value.code == "EXO_PLACED"
    _js(e.value.as_dict())
    assert not launched


def test_run_status_stop_list(launched, lab):
    h = api.run(command="kl", tag="s")
    st = api.run_status(run_id=h.run_id)
    assert st["run_id"] == h.run_id and _js(st)
    assert api.list_runs(n=5)["runs"][0]["run_id"] == h.run_id
    stopped = api.stop(run_id=h.run_id)        # the fake pid 4242 is not alive
    assert stopped["stopped"] is False and _js(stopped)
    with pytest.raises(api.ApiError) as e:
        api.run_status(run_id="nope")
    assert e.value.code == "NOT_FOUND"


def test_fit_geo_build_score_bench_build_argv(launched, lab):
    r = str(lab["root"])
    h = api.fit(kind="moe", src=r, out=r, base=r, vq_layers="0-9",
                opts={"k": 256, "pack": True, "family": "qwen4_exp", "seed": None})
    meta = json.loads((Path(h.run_dir) / "meta.json").read_text())
    assert meta["cmd"] == "fit-moe"
    assert meta["args"] == ["--base", r, "--src", r, "--out", r, "--vq-layers", "0-9",
                            "--k", "256", "--pack", "--family", "qwen4_exp"]
    h = api.fit(kind="dense", src=r, out=r, opts={"layers": "0-3"})
    assert json.loads((Path(h.run_dir) / "meta.json").read_text())["cmd"] == "fit-dense"
    h = api.geo_build(artifact=r, teacher=r, family="qwen4_exp", geomap=r, out=r, reuse=r)
    assert json.loads((Path(h.run_dir) / "meta.json").read_text())["args"][-2:] == ["--reuse", r]
    h = api.score(kind="ppl", args=[r], opts={"rung": ["a", "b"]})
    meta = json.loads((Path(h.run_dir) / "meta.json").read_text())
    assert meta["cmd"] == "score" and meta["args"] == [r, "--rung", "a", "--rung", "b"]
    h = api.bench(kind="active-bytes", args=[r])
    assert json.loads((Path(h.run_dir) / "meta.json").read_text())["cmd"] == "active-bytes"
    assert len(launched) == 5


@pytest.mark.parametrize("call", [
    lambda: api.fit(kind="ple", src="x", out="y"),
    lambda: api.fit(kind="moe", src="x", out="y"),               # no base / vq_layers
    lambda: api.fit(kind="dense", src="x", out="y", base="z"),
    lambda: api.geo_build(artifact="a", teacher="", family="f", geomap="g", out="o"),
    lambda: api.score(kind="speed", opts={}),
    lambda: api.bench(kind="speed-pair"),                         # not MCP-launchable
    lambda: api.score(kind="kl", opts={"--raw": 1}),
    lambda: api.score(kind="kl", opts={"x": {"nested": 1}}),
])
def test_job_argument_validation(launched, call):
    with pytest.raises(api.ApiError) as e:
        call()
    assert e.value.code == "INVALID_ARG"
    assert not launched


# --------------------------------------------------------------------------- #
# Storage and artifacts
# --------------------------------------------------------------------------- #

def test_where_is_and_list_artifacts(lab):
    d = api.where_is(name="Flash-2.1")
    assert d["found"] and d["matches"][0]["path"] == str(lab["art"]) and _js(d)
    assert api.where_is(name="nothing_here")["found"] is False
    with pytest.raises(api.ApiError) as e:
        api.where_is(name="ab")
    assert e.value.code == "INVALID_ARG"
    with pytest.raises(api.ApiError):
        api.where_is(name="abc", kind="weird")
    a = api.list_artifacts()
    assert [x["path"] for x in a["artifacts"]] == [str(lab["art"])] and _js(a)
    with pytest.raises(api.ApiError) as e:
        api.list_artifacts(root="/not/a/root")
    assert e.value.code == "OUTSIDE_ROOTS"


def test_artifact_size_and_info(lab):
    s = api.artifact_size(path=str(lab["art"]))
    assert s["text"] == 2 and s["text_plus_tower_plus_mtp"] == 2
    assert s["index"]["state"] == "ok" and _js(s)
    info = api.artifact_info(path=str(lab["art"]))
    assert info["model_type"] == "qwen4_exp" and info["shards"] == 1 and _js(info)
    for f in (api.artifact_size, api.artifact_info):
        with pytest.raises(api.ApiError) as e:
            f(path=str(lab["tmp"] / "missing"))
        assert e.value.code == "NOT_FOUND"


# --------------------------------------------------------------------------- #
# In-process tools
# --------------------------------------------------------------------------- #

def test_gate_bundle_on_a_synthetic_fixture(lab):
    g = api.gate(check="bundle", artifact=str(lab["art"]))    # no model.py
    assert isinstance(g, api.GateResult)
    assert not g.ok and g.status == "FAIL" and g.exit_code == 1
    assert g.failures == ["FAIL: no bundled model.py"]
    assert g.report == "FAIL: no bundled model.py\n" and _js(g)["ok"] is False


def test_gate_comparator(lab, tmp_path):
    tea = tmp_path / "teacher"
    tea.mkdir()
    _st(tea / "model.safetensors", ["model.layers.0.mlp.weight", "model.embed_tokens.weight"])
    g = api.gate(check="comparator", artifact=str(lab["art"]), teacher=str(tea))
    assert g.ok and g.status == "PASS" and g.failures == []
    _st(tea / "model.safetensors", ["model.layers.0.mlp.weight", "model.embed_tokens.weight",
                                    "model.layers.0.attn.weight"])
    g = api.gate(check="comparator", artifact=str(lab["art"]), teacher=str(tea))
    assert not g.ok and g.status in ("FAIL", "INCONCLUSIVE") and _js(g)
    with pytest.raises(api.ApiError) as e:
        api.gate(check="comparator", artifact=str(lab["art"]))
    assert e.value.code == "INVALID_ARG"
    with pytest.raises(api.ApiError):
        api.gate(check="release", artifact=str(lab["art"]))


def test_gate_cli_output_unchanged(lab, capsys):
    from vqlab.gate import check_bundle
    assert check_bundle.main(["--artifact", str(lab["art"])]) == 1
    assert capsys.readouterr().out == "FAIL: no bundled model.py\n"


def test_kl_pair_in_process(tmp_path):
    np = pytest.importorskip("numpy")
    from safetensors.numpy import save_file
    side = {"kl_cache": {"tokens_sha256": "t", "teacher_head_sha256": "h", "teacher_bytes": 1},
            "numerics": {"mlx": "0"}, "kl_positions": 4}
    for rung, vals in (("arm", [1.0, 2.0, 3.0, 4.0]), ("ref", [2.0, 2.5, 3.5, 5.0])):
        f = tmp_path / f"{rung}__prose.safetensors"
        save_file({"kl_millinats": np.array(vals, dtype=np.float32)}, str(f))
        Path(str(f) + ".json").write_text(json.dumps(side))
    r = api.kl_pair(per_pos_dir=str(tmp_path), arm="arm", ref="ref", corpora=["prose"])
    row = r["corpora"][0]
    assert r["verified"] and row["n"] == 4 and row["delta"] == pytest.approx(-0.75)
    assert _js(r)
    Path(str(tmp_path / "arm__prose.safetensors") + ".json").unlink()
    with pytest.raises(api.ApiError) as e:
        api.kl_pair(per_pos_dir=str(tmp_path), arm="arm", ref="ref", corpora=["prose"])
    assert e.value.code == "FAILED"
    r = api.kl_pair(per_pos_dir=str(tmp_path), arm="arm", ref="ref", corpora=["prose"],
                    allow_unverified=True)
    assert not r["verified"] and r["corpora"][0]["problems"]
    with pytest.raises(api.ApiError):
        api.kl_pair(per_pos_dir=str(tmp_path), arm="arm", ref="ref", corpora="prose")


def test_bundle_runs_the_tool_in_process(lab, monkeypatch):
    from vqlab.bundle import add_model_file, rebundle_dense
    seen = {}

    def fake_main(argv):
        seen["argv"] = argv
        print("wrote model.py")
        return 0
    monkeypatch.setattr(add_model_file, "main", fake_main)
    out = api.bundle(artifact=str(lab["art"]), adopt="all")
    assert out.ok and out.output == "wrote model.py\n" and _js(out)
    assert seen["argv"] == ["--artifact", str(lab["art"]), "--adopt", "all"]
    monkeypatch.setattr(rebundle_dense, "main",
                        lambda argv: (_ for _ in ()).throw(SystemExit("not dense")))
    out = api.bundle(artifact=str(lab["art"]), dense=True)
    assert not out.ok and out.exit_code == 1 and "not dense" in out.output
    with pytest.raises(api.ApiError):
        api.bundle(artifact=str(lab["art"]), runtime="v9")


def test_release_prep(lab, monkeypatch):
    from vqlab.ship import release_prep as rp
    monkeypatch.setattr(rp, "hub_diff", lambda art, repo, h: (["model.py"], [], True))
    (lab["art"] / ".DS_Store").write_text("")
    r = api.release_prep(artifact=str(lab["art"]), repo="owner/name")
    assert r["changed"] == ["model.py"] and r["junk"] == [".DS_Store"]
    assert r["publish_line"].endswith("--repo owner/name --files model.py") and _js(r)
    monkeypatch.setattr(rp, "hub_diff", lambda art, repo, h: ([], [], True))
    assert api.release_prep(artifact=str(lab["art"]), repo="o/n")["publish_line"] is None
    with pytest.raises(api.ApiError):
        api.release_prep(artifact=str(lab["art"]), repo="")


def test_card(lab, monkeypatch, tmp_path):
    from vqlab.ship import card as tool
    seen = {}

    def fake_build(ns):
        seen["ns"] = ns
        return "# Card\n\ncolour\n"
    monkeypatch.setattr(tool, "build", fake_build)
    out = tmp_path / "README.md"
    c = api.card(artifact=str(lab["art"]), kl=["a.json"], this="r1", out=str(out),
                 opts={"base_model": "org/base"})
    assert seen["ns"].kl == ["a.json"] and seen["ns"].base_model == "org/base"
    assert out.read_text() == c.text and c.spellings[0]["word"] == "colour"
    assert not c.releasable and _js(c)["written_to"] == str(out)
    with pytest.raises(api.ApiError):
        api.card(artifact=str(lab["art"]), kl="a.json", this="r1")
    with pytest.raises(api.ApiError):
        api.card(artifact=str(lab["art"]), kl=["a.json"], this="r1", opts={"draft": True})


def test_fits(tmp_path):
    rec = {"fit_id": "f1", "family": "qwen4_exp", "teacher": "Flash-Next", "layer": 3,
           "proj": "gate", "d": 4, "K": 256}
    (tmp_path / "index.jsonl").write_text(json.dumps(rec) + "\n" + json.dumps(
        dict(rec, fit_id="f2", layer=5, K=512)) + "\n")
    assert [r["fit_id"] for r in api.fits(root=str(tmp_path))] == ["f1", "f2"]
    got = api.fits(root=str(tmp_path), teacher="Flash", layers=[5], K=512)
    assert [r["fit_id"] for r in got] == ["f2"] and _js(got)


def test_findings(lab):
    t = api.findings_tail(n=5)
    assert t["entries"][-1]["f"] == 97 and _js(t)
    n = api.next_f_number()
    assert n["next"] == 98 and n["reserved"] is False
    r = api.next_f_number(reserve=True, owner="test")
    assert r["reserved"] and r["next"] == 98
    assert api.next_f_number()["next"] == 99 and _js(r)


def test_version_stamp():
    s = api.version_stamp()
    assert {"mlx", "mlx_lm", "model_type", "runtime", "arch_file"} <= set(s) and _js(s)


def test_dataclasses_are_frozen_json():
    for cls in (api.CommandInfo, api.RunHandle, api.GateResult, api.ToolOutput, api.CardResult):
        assert dataclasses.is_dataclass(cls)
