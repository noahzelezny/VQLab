"""A stale mlx-vlm must not cost a bundle its text model.

The bundle's `_resolve_arch` (runtime/arch_resolve.PRELUDE) prefers mlx_vlm
for a multimodal artifact. An mlx_vlm that is installed but incompatible fails
with ImportError / AttributeError, not ModuleNotFoundError; unless mlx_vlm is
itself the loader, the resolver falls through to mlx_lm and warns that vision
is unavailable.
"""
import types
import warnings

import pytest

from vqlab.runtime import arch_resolve


def _resolver(modules, loader):
    """`_resolve_arch` exactly as bundled, over a fake importlib: `modules`
    maps a dotted name to a module or to the exception importing it raises."""
    src = arch_resolve.PRELUDE.split("\n_MULTIMODAL =")[0]

    def import_module(name):
        got = modules.get(name, ModuleNotFoundError(name))
        if isinstance(got, BaseException):
            raise got
        return got

    ns = {"_importlib": types.SimpleNamespace(import_module=import_module), "_LOADER": loader}
    exec(compile(src, "arch_resolve.PRELUDE", "exec"), ns)
    return ns["_resolve_arch"]


LM = types.ModuleType("mlx_lm.models.gemma4")
VLM = types.ModuleType("mlx_vlm.models.gemma4")


def test_stale_mlx_vlm_falls_back_to_mlx_lm_with_a_warning():
    resolve = _resolver({"mlx_vlm.models.gemma4": ImportError("cannot import name X"),
                         "mlx_lm.models.gemma4": LM}, loader=None)
    with pytest.warns(RuntimeWarning, match="Vision is unavailable"):
        assert resolve("gemma4", True) is LM


def test_mlx_vlm_as_the_loader_still_fails_loudly():
    resolve = _resolver({"mlx_vlm.models.gemma4": ImportError("cannot import name X"),
                         "mlx_lm.models.gemma4": LM}, loader="mlx_vlm")
    with pytest.raises(ImportError):
        resolve("gemma4", True)


def test_a_broken_mlx_lm_arch_is_not_masked():
    resolve = _resolver({"mlx_lm.models.qwen3_5": AttributeError("broken"),
                         "mlx_vlm.models.qwen3_5": VLM}, loader="mlx_lm")
    with pytest.raises(AttributeError):
        resolve("qwen3_5", False)


def test_healthy_mlx_vlm_is_still_preferred_for_multimodal():
    resolve = _resolver({"mlx_vlm.models.gemma4": VLM, "mlx_lm.models.gemma4": LM}, loader=None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve("gemma4", True) is VLM


def test_absent_mlx_vlm_falls_back_silently():
    resolve = _resolver({"mlx_lm.models.gemma4": LM}, loader=None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve("gemma4", True) is LM


def test_neither_runtime_raises_the_last_error():
    resolve = _resolver({}, loader=None)
    with pytest.raises(ModuleNotFoundError):
        resolve("gemma4", True)
