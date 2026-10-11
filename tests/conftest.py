"""Put vqlab's stage folders on sys.path, as `vqlab <cmd>` does, so tests can
import runtime modules by their bare names (vq_switch, vq_dense, ...)."""
import os
import pathlib
import sys

# never create a lab folder in the real home directory (agents/first_run.py);
# set before anything runs, so subprocess `vqlab` calls inherit it too
os.environ["VQLAB_NO_SETUP"] = "1"

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from vqlab import _layout  # noqa: E402,F401

# `import vqlab` no longer installs Knurlogic's architectures; the suite
# loads models the way the CLI does, so it opts in (VQLAB_VENDORED_ARCH=0: stock).
import vqlab  # noqa: E402

vqlab.use_knurlogic_arch()
