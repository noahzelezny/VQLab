"""vqlab mcp — protocol framing, deterministic lookup, run guards, findings ledger.

No GPU, no network: roots and the findings log are redirected to tmp_path.
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import vqlab.mcp_server as m  # noqa: E402


@pytest.fixture
def lab(tmp_path, monkeypatch):
    root = tmp_path / "scratch"
    art = root / "Flash-2.1-v2"
    art.mkdir(parents=True)
    (art / "config.json").write_text(json.dumps({"model_type": "qwen4_exp", "vq": {"d": 4}}))
    (art / "model-00001.safetensors").write_bytes(b"\0" * 4096)
    cache = root / "Exo Models" / "flashnext_teacher_topk_prose"
    cache.mkdir(parents=True)
    monkeypatch.setenv("VQLAB_ROOTS", str(root))
    monkeypatch.setenv("VQLAB_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("VQLAB_GPU_LEASE", str(tmp_path / "gpu.lease"))
    log = tmp_path / "FINDINGS-LOG.md"
    log.write_text("# log\n\n## F97 (2026-09-14) — packing is exact\n\nbody\n\n## F98 (2026-09-14) — corrects\n\nbody\n")
    monkeypatch.setattr(m, "FINDINGS_LOG", log)
    monkeypatch.setattr(m, "_exo_instances", lambda: [])
    return {"root": root, "art": art, "cache": cache, "log": log}


def _rpc(messages):
    inp = io.BytesIO("".join(json.dumps(x) + "\n" for x in messages).encode())
    out = io.BytesIO()
    m.serve(inp, out)
    return [json.loads(l) for l in out.getvalue().decode().splitlines() if l.strip()]


def _text(result):
    return json.loads(result["content"][0]["text"])


def test_handshake_and_tool_list(lab):
    resps = _rpc([
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ])
    assert resps[0]["result"]["protocolVersion"] == m.PROTOCOL_VERSION
    names = {t["name"] for t in resps[1]["result"]["tools"]}
    assert {"where_is", "run", "status", "findings_append"} <= names
    assert "publish" not in names and "publish" not in m.RUN_ALLOWLIST


def test_unknown_method_is_jsonrpc_error(lab):
    (r,) = _rpc([{"jsonrpc": "2.0", "id": 9, "method": "resources/list"}])
    assert r["error"]["code"] == -32601


def test_where_is_finds_the_teacher_cache(lab):
    d = m.t_where_is("flashnext_teacher")
    assert d["found"] and d["matches"][0]["path"] == str(lab["cache"])


def test_where_is_not_found_names_the_roots(lab):
    d = m.t_where_is("nothing_like_this")
    assert d["status"] == "NOT_FOUND" and d["roots_searched"] == [str(lab["root"])]


def test_list_artifacts_and_config(lab):
    a = m.t_list_artifacts()
    assert [x["path"] for x in a["artifacts"]] == [str(lab["art"])]
    c = m.t_artifact_config(str(lab["art"]), keys=["model_type"])
    assert json.loads(c["config"]) == {"model_type": "qwen4_exp"}


def test_config_outside_roots_refused(lab, tmp_path):
    with pytest.raises(m.ToolError) as ei:
        m.t_artifact_config(str(tmp_path))
    assert ei.value.code == "OUTSIDE_ROOTS"


def test_run_guards(lab, tmp_path):
    with pytest.raises(m.ToolError) as e1:
        m.t_run("publish")
    assert e1.value.code == "NOT_ALLOWED"
    with pytest.raises(m.ToolError) as e2:
        m.t_run("score", ["--model", str(tmp_path / "elsewhere")])
    assert e2.value.code == "OUTSIDE_ROOTS"


def test_run_refuses_while_exo_placed(lab, monkeypatch):
    monkeypatch.setattr(m, "_exo_instances", lambda: ["inst-1"])
    with pytest.raises(m.ToolError) as ei:
        m.t_run("score", ["--model", str(lab["art"])])
    assert ei.value.code == "EXO_PLACED"
    # selftest has no model resident and is exempt from the placement check
    monkeypatch.setattr(m, "_lease_holder", lambda: {"holder": "someone"})
    with pytest.raises(m.ToolError) as e2:
        m.t_run("selftest")
    assert e2.value.code == "GPU_BUSY"


def test_run_refuses_when_lease_held(lab, monkeypatch):
    monkeypatch.setattr(m, "_lease_holder", lambda: {"holder": "claude-code-ingest"})
    with pytest.raises(m.ToolError) as ei:
        m.t_run("score", ["--model", str(lab["art"])])
    assert ei.value.code == "GPU_BUSY"


def test_run_launches_detached_and_status_reads_it(lab):
    r = m.t_run("price", ["--help"], tag="t")
    rd = Path(r["run_dir"])
    assert rd.is_dir() and (rd / "meta.json").is_file()
    import time
    for _ in range(40):
        s = m.t_status(r["run_id"])
        if s["status"] in ("completed", "failed", "lost", "deferred"):
            break
        time.sleep(0.5)
    assert s["status"] == "completed" and s["exit_code"] == 0 and s["attempts"] == 1
    assert "--budget-gib" in s["log_tail"]
    assert m.t_list_runs()["runs"][0]["run_id"] == r["run_id"]
    assert m.t_gpu_state()["lease_holder"] is None  # released with the supervisor


def test_findings_numbering_and_append(lab):
    assert m.t_next_f_number()["next"] == 99
    assert [e["f"] for e in m.t_findings_tail(5)["entries"]] == [97, 98]
    out = m.t_findings_append("headline", "art", "vqlab score prose", "prose −2%", "prose +1.05%", "falsified", "body")
    assert out["f"] == 99
    text = lab["log"].read_text()
    assert "## F99 (" in text and "PREDICTION (pre-registered): prose −2%" in text and "VERDICT:    FALSIFIED" in text
    assert m.t_next_f_number()["next"] == 100


def test_findings_append_requires_every_field(lab):
    with pytest.raises(m.ToolError) as e1:
        m.t_findings_append("h", "a", "i", "", "m", "CONFIRMED")
    assert e1.value.code == "INCOMPLETE" and e1.value.extra["missing"] == ["prediction"]
    with pytest.raises(m.ToolError) as e2:
        m.t_findings_append("h", "a", "i", "p", "m", "maybe")
    assert e2.value.code == "BAD_VERDICT"
    with pytest.raises(m.ToolError) as e3:
        m.t_findings_append("h", "a", "i", "p", "m", "CORRECTS")
    assert e3.value.code == "INCOMPLETE"


def test_read_doc_is_confined():
    assert m.t_read_doc("AGENTS.md", 1, 3)["end"] == 3
    with pytest.raises(m.ToolError):
        m.t_read_doc("src/vqlab/cli.py")


def test_call_tool_never_raises():
    r = m.call_tool("where_is", {"bogus": 1})
    assert r["isError"] and _text(r)["error"] == "INVALID_ARG"
    r = m.call_tool("nope", {})
    assert _text(r)["error"] == "UNKNOWN_TOOL"
