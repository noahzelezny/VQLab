"""The namespaced CLI (`vqlab fit moe`) and the flat names it replaces
(`vqlab fit-moe`) are ONE code path: same script, same argv, same build
record, same box guard, same run-log name. Nothing here runs a tool."""
import os
import subprocess
import sys

import pytest

from vqlab import _layout  # noqa: F401
from vqlab import cli

import box_quiet  # noqa: E402
import run_queue as rq  # noqa: E402
import runlog  # noqa: E402

PAIRS = [(flat, cli.SPELLING[flat].split()) for flat in cli.COMMANDS]


@pytest.fixture
def seen(monkeypatch):
    """Run cli.main with the tool, run log, box guard and build record
    stubbed out; record what each would have been handed."""
    got = {}

    def run_path(path, run_name):
        got["script"], got["argv"] = path, list(sys.argv)
    monkeypatch.setattr(cli.runpy, "run_path", run_path)
    monkeypatch.setattr(runlog, "start", lambda cmd, rest: got.setdefault("log", (cmd, rest)))
    monkeypatch.setattr(runlog, "end", lambda run, rc: None)
    monkeypatch.setattr(box_quiet, "guard", lambda cmd: got.setdefault("guard", cmd))
    monkeypatch.setattr(cli, "_record_build", lambda cmd, rest, script: got.setdefault(
        "record", (cmd, rest)))

    def call(argv):
        got.clear()
        monkeypatch.setattr(sys, "argv", ["vqlab", *argv])
        rc = cli.main()
        return rc, dict(got)
    return call


def test_every_command_in_exactly_one_namespace():
    flats = [f for _, subs in cli.NAMESPACES.values() for f in subs.values()]
    assert sorted(flats) == sorted(cli.COMMANDS)
    assert len(set(flats)) == len(flats)


@pytest.mark.parametrize("flat,ns", PAIRS, ids=[p[0] for p in PAIRS])
def test_flat_and_namespaced_dispatch_identically(seen, flat, ns):
    args = ["--out", "/nonexistent/x", "pos"]
    rc_f, a = seen([flat, *args])
    rc_n, b = seen([*ns, *args])
    assert rc_f == rc_n == 0
    assert a == b
    assert a["argv"][1:] == args
    assert a["log"] == (flat, args) and a["guard"] == flat
    assert ("record" in a) == (flat in cli.BUILD_OUTPUTS)
    if flat in cli.BUILD_OUTPUTS:
        assert a["record"] == (flat, args)


def test_resolve_ambiguous_heads():
    # `score` and `bundle` are namespaces AND flat commands
    assert cli.resolve(["score", "ppl", "--x"]) == ("score", ["--x"])
    assert cli.resolve(["score", "--model", "m"]) == ("score", ["--model", "m"])
    assert cli.resolve(["score", "kl-ladder"]) == ("kl-ladder", [])
    assert cli.resolve(["bundle", "moe", "--artifact", "a"]) == ("bundle", ["--artifact", "a"])
    assert cli.resolve(["bundle", "--artifact", "a"]) == ("bundle", ["--artifact", "a"])
    assert cli.resolve(["fit", "nope"]) == (None, ["fit", "nope"])
    assert cli.resolve(["nope"]) == (None, ["nope"])


def _cli(*argv):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        filter(None, [str(_layout.SRC), os.environ.get("PYTHONPATH")]))}
    return subprocess.run([sys.executable, "-m", "vqlab.cli", *argv],
                          capture_output=True, text=True, env=env)


def test_top_help_lists_namespaces_then_every_command():
    p = _cli("--help")
    assert p.returncode == 0
    out = p.stdout
    for ns in cli.NAMESPACES:
        assert f"\n{ns}:\n" in out
    for flat, ns in PAIRS:
        assert " ".join(ns) in out
    assert "[alias: fit-moe]" in out and "[alias: build-dense]" in out
    assert out.index("namespaces") < out.index("\nplan:\n") < out.index("\nlab:\n")


@pytest.mark.parametrize("ns", list(cli.NAMESPACES))
def test_namespace_alone_lists_its_commands(ns):
    p = _cli(ns)
    assert p.returncode == 0
    for sub in cli.NAMESPACES[ns][1]:
        assert f"  {sub} " in p.stdout


def test_unknown_sub_is_refused():
    p = _cli("fit", "nope")
    assert p.returncode == 2 and "unknown command: vqlab fit nope" in p.stderr
    assert _cli("nope").returncode == 2


def test_context_routing_reads_both_spellings():
    assert cli.routed("`fit moe` and `vqlab build pack --out x` and `kl-ladder`") == {
        "fit-moe", "pack", "kl-ladder"}
    assert cli.routed("`score ppl`, `score`, `fit`") == {"score"}


def test_root_context_routes_every_command():
    ctx = (_layout.SRC.parent / "CONTEXT.md").read_text()
    assert not set(cli.COMMANDS) - cli.routed(ctx)


def _q(cmd):
    return {"steps": [{"name": "s", "cmd": cmd, "args": [], "preflight": {"skip": "t"}}]}


@pytest.mark.parametrize("cmd", ["kl-ladder", "fit-moe", "pack", "score", "bundle"])
def test_queue_flat_names_still_validate(cmd):
    q = _q(cmd)
    assert rq.validate(q) == []
    assert q["steps"][0]["cmd"] == cmd


@pytest.mark.parametrize("cmd,flat", [("score kl-ladder", "kl-ladder"), ("fit moe", "fit-moe"),
                                      ("score ppl", "score"), ("bundle moe", "bundle")])
def test_queue_namespaced_names_are_stored_flat(cmd, flat):
    q = _q(cmd)
    assert rq.validate(q) == []
    assert q["steps"][0]["cmd"] == flat      # SCORING / GPU_FREE / RESUMABLE key on it


def test_queue_refuses_unknown_and_namespaced_publish():
    assert any("unknown command" in e for e in rq.validate(_q("fit nope")))
    assert any("human action" in e for e in rq.validate(_q("ship publish")))


def test_renamed_modules_keep_old_names():
    assert _layout.find("vq_397b_codes.py") == _layout.find("fit_moe.py")
    assert _layout.find("pack_artifact.py") == _layout.find("pack.py")
    m = _layout._names()
    assert m["vq_397b_codes"] == m["vqlab.vq_397b_codes"] == "vqlab.fit.fit_moe"
    assert m["pack_artifact"] == m["vqlab.assemble.pack_artifact"] == "vqlab.assemble.pack"
    assert cli.COMMANDS["fit-moe"][0] == "fit_moe.py" and cli.COMMANDS["pack"][0] == "pack.py"
