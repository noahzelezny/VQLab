"""The names other code depends on, frozen while the package is reorganized.

Snapshot (tests/fixtures/public_names.json) of what exists today:
  * every import name _layout answers for (_layout.PERMANENT: the bare
    RUNTIME stems and the `vqlab.*` spellings shipped text names);
  * every old FILE name _layout.find() maps (archived queue files name
    files such as vq_397b_codes.py); these are not import names;
  * every CLI command;
  * the MCP `run` allowlist and its GPU-free subset.

Names may be ADDED; none may disappear. The MCP sets must match exactly:
an agent silently gaining (or losing) a runnable command is a policy change,
not a refactor. Regenerate after a deliberate change with

    PYTHONPATH=src python tests/test_public_imports.py --write
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "tests" / "fixtures" / "public_names.json"

sys.path.insert(0, str(ROOT / "src"))
from vqlab import _layout  # noqa: E402
from vqlab import cli  # noqa: E402

RUNTIME_STEMS = sorted(p.stem for p in (ROOT / "src/vqlab/runtime").glob("*.py")
                       if p.stem != "__init__")


def _mcp():
    from vqlab.ops import runs     # the MCP `run` tool's allowlist lives here
    return runs


def _current():
    m = _mcp()
    return {"import_names": sorted(_layout.PERMANENT),
            "file_names": sorted(_layout.FILE_NAMES),
            "commands": sorted(cli.COMMANDS),
            "mcp_run_allowlist": sorted(m.RUN_ALLOWLIST),
            "mcp_gpu_free": sorted(m.GPU_FREE)}


@pytest.fixture(scope="module")
def snap():
    return json.loads(SNAPSHOT.read_text())


def test_no_import_name_disappears(snap):
    gone = sorted(set(snap["import_names"]) - set(_layout.PERMANENT))
    assert not gone, f"import names no longer resolved: {gone}"


def test_every_import_name_resolves(snap):
    missing = [n for n in snap["import_names"] if importlib.util.find_spec(n) is None]
    assert not missing, missing


def test_every_old_file_name_finds_a_file(snap):
    missing = [s for s in snap["file_names"] if not _layout.find(f"{s}.py").is_file()]
    assert not missing, f"old file names no longer found: {missing}"


# Bare names an app might own. `import vqlab` must not answer for any of them
# (R1: `import settings` after `import vqlab` gave vqlab's module).
FOREIGN_BARE = [
    "settings", "registry", "hub", "exo", "power", "pin", "mix", "pack",
    "card", "size", "price", "doctor", "gui", "parity", "scratch", "serve",
    "publish", "fits", "artifact", "smoke", "families", "provenance",
    "kmeans", "config", "commands", "api", "ops", "queue", "util", "core",
    "build", "fit", "plan", "score", "bench", "gate", "ship", "records",
    "agents", "family", "mtp", "selftest", "skipzero", "bundle", "runtime",
]


def test_import_vqlab_answers_for_no_foreign_bare_name():
    code = (
        "import importlib, importlib.util, json, pathlib, vqlab, vqlab.api\n"
        "pkg = pathlib.Path(vqlab.__file__).resolve().parent\n"
        "bad = []\n"
        f"for n in {FOREIGN_BARE!r}:\n"
        "    s = importlib.util.find_spec(n)\n"
        "    o = s and s.origin and pathlib.Path(s.origin).resolve()\n"
        "    if s is not None and (not o or pkg in o.parents):\n"
        "        bad.append(n)\n"
        f"stems = {RUNTIME_STEMS!r}\n"
        "ok = [importlib.import_module(s) is importlib.import_module('vqlab.runtime.' + s)\n"
        "      for s in stems]\n"
        "print(json.dumps({'bad': bad, 'ok': all(ok)}))\n")
    with tempfile.TemporaryDirectory() as d:   # cwd holds no stray modules
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           cwd=d, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out == {"bad": [], "ok": True}, out


def test_bare_names_are_only_runtime_stems():
    bare = {k for k in _layout.PERMANENT if "." not in k}
    assert bare == set(RUNTIME_STEMS), sorted(bare ^ set(RUNTIME_STEMS))


@pytest.mark.parametrize("stem", RUNTIME_STEMS)
def test_runtime_aliases_are_one_module(stem):
    canon = importlib.import_module(f"vqlab.runtime.{stem}")
    assert importlib.import_module(f"vqlab.{stem}") is canon
    assert importlib.import_module(stem) is canon
    assert canon.__name__ == f"vqlab.runtime.{stem}"


def test_reload_by_bare_name_keeps_identity():
    import vq_switch as bare
    import vqlab.runtime.vq_switch as canon
    import vqlab.vq_switch as old
    assert bare is canon is old
    assert importlib.reload(bare) is canon and canon.__name__ == "vqlab.runtime.vq_switch"


def test_no_command_disappears(snap):
    gone = sorted(set(snap["commands"]) - set(cli.COMMANDS))
    assert not gone, f"CLI commands removed: {gone}"


def test_mcp_sets_match_snapshot(snap):
    m = _mcp()
    assert sorted(m.RUN_ALLOWLIST) == snap["mcp_run_allowlist"]
    assert sorted(m.GPU_FREE) == snap["mcp_gpu_free"]


def test_publish_is_never_mcp_runnable():
    assert "publish" not in _mcp().RUN_ALLOWLIST


def test_every_command_help_exits_zero(tmp_path):
    """`vqlab <cmd> --help` for every command, through the real dispatcher.
    Run log goes to a temp dir; --help never takes the box_quiet guard."""
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), VQLAB_LOG_DIR=str(tmp_path))

    def run(cmd):
        p = subprocess.run([sys.executable, "-m", "vqlab.cli", cmd, "--help"],
                           capture_output=True, text=True, env=env, timeout=600)
        return cmd, p.returncode, (p.stdout + p.stderr)[-400:]

    with ThreadPoolExecutor(max(2, min(8, (os.cpu_count() or 2)))) as ex:
        bad = [r for r in ex.map(run, sorted(cli.COMMANDS)) if r[1] != 0]
    assert not bad, "\n".join(f"{c}: rc={rc}\n{out}" for c, rc, out in bad)


if __name__ == "__main__" and "--write" in sys.argv:
    SNAPSHOT.parent.mkdir(exist_ok=True)
    SNAPSHOT.write_text(json.dumps({"_doc": "names other code depends on "
                                    "(tests/test_public_imports.py).", **_current()},
                                   indent=1) + "\n")
    print(f"wrote {SNAPSHOT}")
