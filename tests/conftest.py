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


ROOT = pathlib.Path(__file__).resolve().parents[1]


def pytest_collection_modifyitems(config, items):
    """A `checkout` test needs the git tree (root docs, `git ls-files`, the
    GUI's source, the ratchet's linters); an unpacked sdist has none."""
    if (ROOT / ".git").exists():
        return
    skip = pytest.mark.skip(reason="needs the source checkout, not an sdist")
    for item in items:
        if item.get_closest_marker("checkout"):
            item.add_marker(skip)


@pytest.fixture
def lab_dirs(tmp_path, monkeypatch):
    """Every lab location these tests touch, each an empty folder under
    tmp_path: .records, .queues, .scratch, .teachers, .families, .models and
    .hf_cache (and .root)."""
    env = {"records": "VQLAB_RECORDS", "queues": "VQLAB_QUEUE_DIR", "scratch": "VQLAB_SCRATCH",
           "teachers": "VQLAB_TEACHERS_DIR", "families": "VQLAB_FAMILIES_DIR",
           "models": "VQLAB_MODELS_DIR", "hf_cache": "HF_HUB_CACHE"}
    dirs = {}
    for name, var in env.items():
        dirs[name] = tmp_path / name
        dirs[name].mkdir()
        monkeypatch.setenv(var, str(dirs[name]))
    return types.SimpleNamespace(root=tmp_path, **dirs)
