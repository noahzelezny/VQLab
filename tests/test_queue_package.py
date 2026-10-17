"""vqlab.queue: the stdlib `queue` stays the stdlib's in a vqlab process."""
import json
import subprocess
import sys

from vqlab import _layout


def test_import_queue_is_the_stdlib_in_a_vqlab_process():
    """`import queue` after loading vqlab, the CLI and every queue module must
    be the standard library's; concurrent.futures (which imports it) works."""
    code = (
        "import vqlab, vqlab._layout, vqlab.cli, vqlab.agents.mcp.server\n"
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
    names = _layout.PERMANENT
    assert "queue" not in names and "vqlab.queue" not in names
    assert not any(v.startswith("vqlab.queue") for v in names.values())
    import queue
    assert queue.__name__ == "queue" and not hasattr(queue, "runner")


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


def test_status_line_for_a_step_that_never_ran():
    """A step skipped after an earlier failure has attempts None; status must
    still print it (it crashed on `None > 1`)."""
    from vqlab.queue import report
    row = {"status": "fail", "name": "vision", "seconds": None, "attempts": None}
    line = report._step_line(2, row, {"eta_s": None, "basis": ""})
    assert "vision" in line and "x" not in line.split("vision")[1]
