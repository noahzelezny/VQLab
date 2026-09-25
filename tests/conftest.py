"""Put vqlab's stage folders on sys.path, as `vqlab <cmd>` does, so tests can
import runtime modules by their bare names (vq_switch, vq_dense, ...)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from vqlab import _layout  # noqa: E402,F401
