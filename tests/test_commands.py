"""vqlab/commands.py: the one command registry the CLI, MCP and queue read."""
import ast
import importlib.util
import json
import pathlib

from vqlab import cli, commands

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_registry_is_pure_data():
    tree = ast.parse((ROOT / "src/vqlab/commands.py").read_text())
    mods = {n.module if isinstance(n, ast.ImportFrom) else a.name
            for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
            for a in n.names}
    assert mods <= {"__future__", "typing"}, mods


def test_every_module_exists_and_names_its_file():
    for c in commands.COMMANDS.values():
        spec = importlib.util.find_spec(c.module)
        # `file` is relative to the package folder; a module inside a
        # sub-package (vqlab.agents.mcp.server) names it with its folder
        assert spec is not None and spec.origin.endswith("/" + c.file), c
        assert c.module.replace(".", "/").endswith(c.file[:-3].removesuffix("/__init__")), c
        assert c.name in commands.NAMESPACES[c.stage][1].values(), c


def test_every_command_in_exactly_one_namespace():
    flats = [f for _, subs in commands.NAMESPACES.values() for f in subs.values()]
    assert sorted(flats) == sorted(commands.COMMANDS)


def test_cli_views_are_derived():
    assert cli.COMMANDS is commands.COMMANDS
    assert cli.resolve is commands.resolve and cli.NAMESPACES is commands.NAMESPACES


def test_mcp_sets_come_from_the_registry_and_match_the_snapshot():
    from vqlab.ops import runs as m
    snap = json.loads((ROOT / "tests/fixtures/public_names.json").read_text())
    assert m.RUN_ALLOWLIST == commands.mcp_allowlist()
    assert m.GPU_FREE == commands.mcp_gpu_free()
    assert sorted(m.RUN_ALLOWLIST) == snap["mcp_run_allowlist"]
    assert sorted(m.GPU_FREE) == snap["mcp_gpu_free"]


def test_publish_is_never_mcp():
    assert commands.COMMANDS["publish"].mcp is False
