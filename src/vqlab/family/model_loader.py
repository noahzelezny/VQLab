#!/usr/bin/env python3
"""Runtime dispatch: which library loads a family's models, in one place.

Design decision (2026-08-29): provision for EITHER mlx-lm or mlx-vlm
rather than forking mlx-lm. Each FAMILY entry may carry a `runtime` field
("mlx_lm" default, or "mlx_vlm"); stream_convert, stream_score and smoke
route every model load through load_for_family(), so when mlx-lm eventually
gains a family's class the switch is ONE registry field — no call sites
change, and the fork option stays dead.

Why glm5_next needs this: mlx-lm has no glm5_next class (checked
2026-08-29) but mlx-vlm ships one (PR #2030, fa27a9a, landed 08-26).
Both runtimes honour the in-checkpoint `model_file` bundle mechanism —
verified by reading mlx-vlm's utils.py load path (raw fetch, 08-29): it
imports the model module from the file named in config when present, same
as mlx_lm — so bundled-runtime artifacts serve under either. PROVISIONAL
until executed: no mlx_vlm import has ever run in this repo's venvs.

III.13 applies to every load made through here: the RESOLVED module path,
not the intended one, is what a runtime-dependent claim must name — call
resolved_runtime_note(model) and print it.

CPU-only checks of the dispatch logic (loads nothing):

    pytest tests/test_model_loader.py
"""
from __future__ import annotations

import os
import pathlib
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

import mlx.core as mx

from vqlab.family.registry import FAMILY

if TYPE_CHECKING:
    from mlx import nn
    from mlx_lm.tokenizer_utils import TokenizerWrapper

RUNTIMES = ("mlx_lm", "mlx_vlm")


def runtime_for(family: str) -> str:
    """The runtime a family loads under. Unknown/dense families default to
    mlx_lm — the behaviour every existing call site had before this file."""
    entry = FAMILY.get(family, {})
    rt = entry.get("runtime", "mlx_lm")
    # Knurlogic serves some architectures that used to exist only in mlx_vlm
    # (glm5_next) under mlx_lm.models, and VQLab loads Knurlogic's files
    # (vqlab.core.box.arch_finder). When it serves this family's model_type, load
    # through mlx_lm: that is the arithmetic users run, and a VQLab env has
    # no mlx_vlm (2026-10-03: GLM could not load in ~/.venvs/vqlab at all).
    if rt == "mlx_vlm" and entry.get("model_type"):
        from vqlab.core.box import arch_finder as arch
        if arch.enabled() and entry["model_type"] in arch.available():
            rt = "mlx_lm"
    if rt not in RUNTIMES:
        raise ValueError(f"family {family!r} declares unknown runtime {rt!r}")
    return rt


def family_for_model_type(model_type: str) -> str | None:
    """Resolve a config's model_type to a FAMILY name, for call sites that
    only have an artifact (smoke, stream_score). Entries opt in by carrying
    a `model_type` field; families without one are unreachable this way,
    which is deliberate — guessing a family from a name is how the wrong
    --family gate ran on 08-21."""
    hits = [f for f, e in FAMILY.items() if e.get("model_type") == model_type]
    if len(hits) > 1:
        raise ValueError(f"model_type {model_type!r} claimed by {hits}")
    return hits[0] if hits else None


def load_for_family(family: str, path, lazy: bool = True,
                    cpu_stream: bool = False):
    """Load (model, config) via the family's declared runtime.

    mlx_lm  : mlx_lm.utils.load_model — returns (model, config), honours
              in-checkpoint model_file bundles (trust_remote_code).
    mlx_vlm : mlx_vlm.utils.load_model + load_config — load_model returns
              the bare nn.Module (vision tower INCLUDED; text stack at
              model.language_model.model.layers), config loaded separately.
              PROVISIONAL: written from a source read of mlx-vlm main,
              never yet executed here.
    """
    path = pathlib.Path(path)
    rt = runtime_for(family)
    # CPU-STREAM LOAD IS OPT-IN, AND IT IS NOT NEUTRAL (measured 2026-09-17).
    # Weight reads bind to whatever stream is current when they are CREATED;
    # a later `with mx.stream(mx.cpu): mx.eval(...)` does NOT rebind them, so
    # a lazily-loaded block "evaluated on the CPU stream" still issues a METAL
    # command buffer. A block big enough for that read to outlast the watchdog
    # dies inside it: the 751 GiB 397B teacher (12.2 GiB blocks) killed every
    # streamed pass in eval_params_budgeted with a GPU Timeout, one layer
    # further each run as the page cache warmed. Loading under the CPU stream
    # fixes it -- 14 layers straight through at a steady 9 s/layer.
    #
    # But it CHANGES NUMBERS on some families, so it cannot be a global
    # default. Same artifact, same cache, Flash-3.2 prose at 12288:
    #     without  ppl 5.000427  KL 156.7034
    #     with     ppl 5.025818  KL 155.1233
    # while qwen3_5_moe (35B-3.4 @12288) reads 5.414175 either way. CPU and
    # GPU kernels round differently, and every published qwen4_exp number
    # used the GPU-stream path -- one harness (house rule) means that family
    # keeps it. Callers that need the CPU-stream load ask for it explicitly.
    if cpu_stream:
        with mx.stream(mx.cpu):
            return _load_on_current_stream(family, path, rt, lazy)
    return _load_on_current_stream(family, path, rt, lazy)


def install_vendored_arch() -> bool:
    """Install Knurlogic's architecture finder; True when it serves.

    Every load routes through here, so which arch file serves a model
    must not depend on how the program was invoked. vqlab/__init__ only
    installs the finder when sys.argv names a vqlab entry point INSIDE the
    imported package: a copy of the script (a worktree pin, a scratch
    copy) or an interpreter without knurlogic silently loaded stock
    mlx-lm, whose qwen3_5 sanitize folds the norms' +1 in bf16 where
    Knurlogic's (edit 8) keeps them float32 -- the same convert wrote 61
    norm weights in two dtypes (2026-10-13)."""
    from vqlab.core.box import arch_finder
    return arch_finder.install_arch_finder()


def require_vendored_arch() -> None:
    """Refuse to run with stock mlx-lm by accident: knurlogic missing
    and VQLAB_VENDORED_ARCH not set to 0. A writer (stream-convert) calls
    this; a stock build stays possible, but only when asked for."""
    from vqlab.core.box import arch_finder
    if arch_finder.enabled() and not install_vendored_arch():
        raise SystemExit(
            "FAIL: knurlogic is not importable in this interpreter, so stock "
            "mlx-lm would serve the architecture and the output's arithmetic "
            "and dtypes would differ from a vqlab-venv build. Use the vqlab "
            "venv, or set VQLAB_VENDORED_ARCH=0 to build on stock mlx-lm "
            "deliberately.")


def trust_remote_code_kwargs(loader: Callable[..., object]) -> dict[str, object]:
    """The keyword that lets this mlx-lm build run an artifact's bundled
    model.py, for `loader` (mlx_lm.utils.load or load_model).

    mlx-lm >= 0.32 refuses an in-checkpoint model.py unless
    trust_remote_code=True is passed; some older builds took it through
    model_config instead, and the grafted exo runtime takes neither (it
    raised TypeError and broke layer-leverage, 2026-09-13). The signature
    is probed, never the version; the one home for this check."""
    import inspect
    try:
        params = inspect.signature(loader).parameters
    except (TypeError, ValueError):
        return {}
    if "trust_remote_code" in params:
        return {"trust_remote_code": True}
    if "model_config" in params:
        return {"model_config": {"trust_remote_code": True}}
    return {}


def load_artifact(path: str | os.PathLike, lazy: bool = False, strict: bool = True,
                  vendored_arch: bool = True) -> tuple[nn.Module, TokenizerWrapper]:
    """Load (model, tokenizer) from an artifact dir through mlx-lm, the way
    the CLI tools need it: bundled model.py trusted, Knurlogic's arch
    served.

    strict=False drops checkpoint tensors the model class does not build
    (load_model + load_tokenizer; mlx_lm.load has no such switch) -- only
    for tensors confirmed dead. vendored_arch=False leaves the import
    machinery alone, for an instrument that chooses its arch source itself
    (runtime-equiv)."""
    from mlx_lm.utils import load, load_model, load_tokenizer
    if vendored_arch:
        install_vendored_arch()
    if strict:
        return load(str(path), lazy=lazy, **trust_remote_code_kwargs(load))
    path = pathlib.Path(path)
    model, _ = load_model(path, lazy=lazy, strict=False,
                          **trust_remote_code_kwargs(load_model))
    return model, load_tokenizer(path)


def _load_on_current_stream(family, path, rt, lazy):
    install_vendored_arch()
    if rt == "mlx_lm":
        from mlx_lm.utils import load_model
        return load_model(path, lazy=lazy, **trust_remote_code_kwargs(load_model))
    # mlx_vlm
    from mlx_vlm.utils import load_config, load_model
    config = load_config(path)
    model = load_model(path, lazy=lazy)
    return model, config


def resolved_runtime_note(model) -> str:
    """One line naming the module file the model class ACTUALLY came from
    (III.13: instrument the import, never assume which copy runs)."""
    cls = type(model)
    mod = sys.modules.get(cls.__module__)
    path = getattr(mod, "__file__", f"<module {cls.__module__!r} — no file>")
    return f"resolved runtime: {cls.__name__} <- {path}"


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        # moved to tests/test_model_loader.py (untangle step 14)
        raise SystemExit("model_loader's self-check is tests/test_model_loader.py: "
                         "run `pytest tests/test_model_loader.py`")
    raise SystemExit(__doc__)
