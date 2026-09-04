"""Pin that drafting on a glm5 family installs the absorbed-MLA shim.

The regression this guards: the GLM card said "vqlab serve installs it",
but no serve/generate path ever called glm5_shim.install(). Without the
shim, the L=2 MTP verify forward pays the unabsorbed latent-cache
expansion (measured 23-40x per layer at long Kv) and the sidecar is a net
loss. The install now lives in load_mtp_head — the one choke point every
drafting entry (`vqlab serve`, `vqlab mtp-generate`, `vqlab mtp-bench`)
goes through — so this test pins that seam, not any single CLI.
"""
import sys

import pytest

pytest.importorskip("mlx.core")

from vqlab.mtp import loop
from vqlab.mtp.registry import FamilySpec, register, unregister

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import toy_family  # noqa: E402


@pytest.fixture
def recorded_install(monkeypatch):
    """Stub vqlab.glm5_shim.install so no mlx_vlm import is needed."""
    from vqlab import glm5_shim
    calls = []
    monkeypatch.setattr(glm5_shim, "install", lambda *a, **k: calls.append(1) or True)
    return calls


def test_maybe_install_fires_for_glm5_families_only(recorded_install):
    def spec(name):
        return FamilySpec(name=name, head="x:y", capture="z", draft_cache=None)

    for name in loop._GLM5_FAMILIES:
        assert loop._maybe_install_glm5_shim(spec(name)) is True
    assert len(recorded_install) == len(loop._GLM5_FAMILIES)

    assert loop._maybe_install_glm5_shim(spec("qwen4_exp")) is False
    assert len(recorded_install) == len(loop._GLM5_FAMILIES)


def test_load_mtp_head_installs_shim_for_glm5(tmp_path, monkeypatch,
                                              recorded_install):
    """Full wiring: a glm5-named family with a present sidecar reaches the
    shim install through load_mtp_head itself."""
    spec = FamilySpec(name="glm5_next",
                      head="toy_family:ToyHead",
                      capture="hyper_connection_mixer",
                      draft_cache="ToyDraftCache",
                      sidecar_name="toy-head.safetensors",
                      cache_semantics="reassign")
    register(spec, replace=True)
    try:
        monkeypatch.setattr(toy_family.ToyHead, "from_sidecar",
                            classmethod(lambda cls, m, a, s: cls()),
                            raising=False)
        (tmp_path / spec.sidecar_name).touch()
        model = toy_family.ToyModel() if hasattr(toy_family, "ToyModel") else None
        if model is None:
            pytest.skip("toy_family exposes no ToyModel")
        head, got = loop.load_mtp_head(model, family="glm5_next",
                                       model_path=tmp_path)
        assert got.name == "glm5_next"
        assert recorded_install, ("load_mtp_head resolved a glm5 family and "
                                  "found the sidecar but never installed the "
                                  "absorbed-MLA shim")
    finally:
        unregister("glm5_next")
        # re-register the real glm5 specs the module import set up
        import importlib
        importlib.reload(sys.modules["vqlab.mtp.registry"])


def test_load_mtp_head_leaves_other_families_alone(tmp_path, monkeypatch,
                                                   recorded_install):
    toy_family.install()
    try:
        monkeypatch.setattr(toy_family.ToyHead, "from_sidecar",
                            classmethod(lambda cls, m, a, s: cls()),
                            raising=False)
        (tmp_path / toy_family.SPEC.sidecar_name).touch()
        model = toy_family.ToyModel() if hasattr(toy_family, "ToyModel") else None
        if model is None:
            pytest.skip("toy_family exposes no ToyModel")
        loop.load_mtp_head(model, family=toy_family.FAMILY,
                           model_path=tmp_path)
        assert not recorded_install
    finally:
        toy_family.remove()
