"""vqlab.queue (untangle step 12): the stdlib `queue` stays the stdlib's, and
every old queue name still answers for one release."""
import importlib
import json
import subprocess
import sys
import warnings

from vqlab import _layout


def test_import_queue_is_the_stdlib_in_a_vqlab_process():
    """`import queue` after loading vqlab, the CLI and every queue module must
    be the standard library's; concurrent.futures (which imports it) works."""
    code = (
        "import vqlab, vqlab._layout, vqlab.cli, vqlab.agents.mcp_server\n"
        "import vqlab.queue.runner, vqlab.queue.steps, vqlab.queue.files\n"
        "import vqlab.queue.eta, vqlab.queue.validate, vqlab.queue.report\n"
        "import queue, sysconfig, pathlib, json\n"
        "from concurrent.futures import ThreadPoolExecutor\n"
        "stdlib = pathlib.Path(sysconfig.get_paths()['stdlib']).resolve()\n"
        "f = pathlib.Path(queue.__file__).resolve()\n"
        "q = queue.Queue(); q.put(1)\n"
        "with ThreadPoolExecutor(2) as ex: r = list(ex.map(abs, [-1, -2]))\n"
        "print(json.dumps({'file': str(f), 'in_stdlib': f.parent == stdlib,\n"
        "                  'name': queue.__name__, 'got': q.get(), 'pool': r,\n"
        "                  'simple': hasattr(queue, 'SimpleQueue')}))\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={"PYTHONPATH": str(_layout.SRC), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["in_stdlib"] and out["name"] == "queue" and out["simple"], out
    assert out["got"] == 1 and out["pool"] == [1, 2]


def test_no_bare_queue_alias():
    names = _layout._names()
    assert "queue" not in names and "vqlab.queue" not in names
    assert not any(v == "vqlab.queue" for v in names.values())
    # queue/'s stems are too generic to be bare names
    assert not any(v.startswith("vqlab.queue.") and "." not in k for k, v in names.items()
                   if k not in ("queue_eta", "validate_queue", "reserve"))
    import queue
    assert queue.__name__ == "queue" and not hasattr(queue, "runner")


def test_moved_names_are_one_module_and_warn():
    for old, new in (("vqlab.agents.queue_eta", "vqlab.queue.eta"),
                     ("vqlab.gate.validate_queue", "vqlab.queue.validate"),
                     ("vqlab.agents.reserve", "vqlab.queue.reserve"),
                     ("queue_eta", "vqlab.queue.eta"),
                     ("validate_queue", "vqlab.queue.validate")):
        sys.modules.pop(old, None)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mod = importlib.import_module(old)
        assert mod is importlib.import_module(new), old
        assert any(issubclass(x.category, DeprecationWarning) for x in w), old


def test_run_queue_reexports_every_old_name():
    """agents/run_queue.py was split three ways; the old module still
    exposes every public name it had and the private ones tests used."""
    sys.modules.pop("vqlab.agents.run_queue", None)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rq = importlib.import_module("vqlab.agents.run_queue")
    assert any(issubclass(x.category, DeprecationWarning) for x in w)
    from vqlab.queue import files, remote, report, runner, steps
    old = ["REPO", "SCHEMA", "SCORING", "OUT_FLAGS", "PREFLIGHT_MARGIN", "TERMINAL",
           "preflight_root", "queues_dir", "validate", "installed", "create", "static_check",
           "pin_check", "run", "status", "remote_paths", "remote_check", "run_on", "wait",
           "main", "_now", "_slug", "_git", "_save", "_needs_gpu", "_package_hash",
           "_NAMED", "_PATHLIKE", "_strip_name", "_path_args", "_input_bytes", "_stop",
           "_term", "_argv", "_env", "_redirect_outs", "_run_step", "_latest",
           "_REMOTE_PROBE", "_mount", "queue_eta"]
    missing = [n for n in old if not hasattr(rq, n)]
    assert not missing, missing
    assert rq.SCHEMA == "vqlab.queue/1"
    assert rq.main is runner.main and rq.validate is steps.validate
    assert rq.create is files.create and rq.wait is report.wait and rq.run_on is remote.run_on
    assert rq._slug("a b") == "a_b"


def test_commands_point_at_the_queue_package():
    from vqlab import commands
    assert commands.COMMANDS["queue"].module == "vqlab.queue.runner"
    assert commands.COMMANDS["validate"].module == "vqlab.queue.validate"
    assert commands.COMMANDS["reserve"].module == "vqlab.queue.reserve"
    for c in ("queue", "validate", "reserve"):
        assert _layout.find(commands.COMMANDS[c].file).is_file()


def test_queue_never_imports_tools_or_agents():
    """L11: the queue launches commands BY NAME (vqlab.commands); it imports
    nothing from agents / selftest / cli (checked statically)."""
    import ast
    bad = []
    for f in (_layout.PKG / "queue").glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            mods = ([a.name for a in node.names] if isinstance(node, ast.Import)
                    else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            bad += [f"{f.name}: {m}" for m in mods
                    if m.split(".")[:2] in (["vqlab", "agents"], ["vqlab", "cli"],
                                            ["vqlab", "selftest"], ["vqlab", "api"])]
    assert not bad, bad
