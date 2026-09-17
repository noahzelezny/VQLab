#!/usr/bin/env python3
"""Runtime dispatch: which library loads a family's models, in one place.

Noah's design ruling (2026-08-29): provision for EITHER mlx-lm or mlx-vlm
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

CPU-only selftest (dispatch logic only, loads nothing):

    python -m vqlab.runtime_load --selftest
"""
from __future__ import annotations

import pathlib
import mlx.core as mx
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from families import FAMILY

RUNTIMES = ("mlx_lm", "mlx_vlm")


def runtime_for(family: str) -> str:
    """The runtime a family loads under. Unknown/dense families default to
    mlx_lm — the behaviour every existing call site had before this file."""
    rt = FAMILY.get(family, {}).get("runtime", "mlx_lm")
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


def _load_on_current_stream(family, path, rt, lazy):
    if rt == "mlx_lm":
        import inspect
        from mlx_lm.utils import load_model
        # `trust_remote_code` was REMOVED from mlx-lm's load_model signature
        # (present in older builds, absent in the grafted/exo runtime as of
        # 2026-09-13, where it raised TypeError and broke layer-leverage).
        # Pass it only when the installed build accepts it; arches the
        # runtime implements natively (e.g. qwen4_exp) need no remote code.
        kwargs = {"lazy": lazy}
        if "trust_remote_code" in inspect.signature(load_model).parameters:
            kwargs["trust_remote_code"] = True
        return load_model(path, **kwargs)
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


def _selftest() -> int:
    ok = True

    def report(name, good, detail=""):
        nonlocal ok
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {name}"
              f"{(' — ' + detail) if detail else ''}")

    report("qwen4_exp -> mlx_lm", runtime_for("qwen4_exp") == "mlx_lm")
    report("qwen3_5 (no field) -> mlx_lm", runtime_for("qwen3_5") == "mlx_lm")
    report("glm5_next -> mlx_vlm", runtime_for("glm5_next") == "mlx_vlm")
    report("unknown family -> mlx_lm default",
           runtime_for("no_such_family") == "mlx_lm")
    report("model_type glm5_next -> family",
           family_for_model_type("glm5_next") == "glm5_next")
    report("unclaimed model_type -> None",
           family_for_model_type("qwen3_5_moe") is None)
    # both directions (III.5): a bad runtime value must raise, not default
    saved = FAMILY.get("glm5_next", {}).get("runtime")
    try:
        FAMILY["glm5_next"]["runtime"] = "torch"
        try:
            runtime_for("glm5_next")
            report("bad runtime value FAILS", False, "accepted 'torch'")
        except ValueError:
            report("bad runtime value FAILS", True)
    finally:
        FAMILY["glm5_next"]["runtime"] = saved
    print("all checks passed" if ok else "CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    raise SystemExit(__doc__)
