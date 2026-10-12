"""Put vqlab's stage folders on sys.path, as `vqlab <cmd>` does, so tests can
import runtime modules by their bare names (vq_switch, vq_dense, ...), and
the shared `lab_dirs` fixture."""
import os
import pathlib
import sys
import types

import pytest

# never create a lab folder in the real home directory (agents/first_run.py);
# set before anything runs, so subprocess `vqlab` calls inherit it too
os.environ["VQLAB_NO_SETUP"] = "1"

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from vqlab import _layout  # noqa: E402,F401

# `import vqlab` no longer installs Knurlogic's architectures; the suite
# loads models the way the CLI does, so it opts in (VQLAB_VENDORED_ARCH=0: stock).
import vqlab  # noqa: E402

vqlab.use_knurlogic_arch()


@pytest.fixture
def lab_dirs(tmp_path, monkeypatch):
    """Every lab location these tests touch, each an empty folder under
    tmp_path: .records, .queues, .scratch, .teachers, .families (and .root)."""
    env = {"records": "VQLAB_RECORDS", "queues": "VQLAB_QUEUE_DIR", "scratch": "VQLAB_SCRATCH",
           "teachers": "VQLAB_TEACHERS_DIR", "families": "VQLAB_FAMILIES_DIR"}
    dirs = {}
    for name, var in env.items():
        dirs[name] = tmp_path / name
        dirs[name].mkdir()
        monkeypatch.setenv(var, str(dirs[name]))
    return types.SimpleNamespace(root=tmp_path, **dirs)
