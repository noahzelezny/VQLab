"""gate/mtp_smoke_head: the trunk loads through family.model_loader."""
import pytest

from vqlab.family import model_loader
from vqlab.gate import mtp_smoke_head


class _Loaded(Exception):
    pass


def test_trunk_loads_lazily_through_load_artifact(monkeypatch):
    # 2026-10-15: a bare mlx_lm load() without trust_remote_code was refused
    # on every bundled VQ artifact, failing each MTP build queue at this step
    calls = []

    def fake(path, lazy=False):
        calls.append((path, lazy))
        raise _Loaded

    monkeypatch.setattr(model_loader, "load_artifact", fake)
    with pytest.raises(_Loaded):
        mtp_smoke_head.main(["--model", "art", "--sidecar", "s.safetensors"])
    assert calls == [("art", True)]
