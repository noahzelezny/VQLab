"""Per-family registration for MTP speculative decoding.

Adding a family is a table entry here plus a head module. Nothing else in the
package names an architecture; the loop, the caches and the sampler are all
family-agnostic.

A `FamilySpec` says four things, and every one of them is a place where
architectures genuinely differ:

  head             where the drafting head lives ("module:Class"). The class
                   must expose `from_sidecar(model, arch, path)` and
                   `draft_logits(h_row, next_ids, cache)`.
  capture          dotted path, relative to the trunk core, of the submodule
                   whose INPUT is the pre-lm_head activation the head drafts
                   from. There is no public mlx-lm hook for this, so we wrap
                   that one module for the duration of the generation (see
                   capture.py) rather than monkeypatching the class.
  draft_cache      the attribute on the architecture module that builds the
                   head's own KV cache.
  cache_semantics  "reassign" or "copy" — see caches.py. qwen4_exp reassigns
                   its recurrent cache slots rather than mutating them, which
                   makes snapshots free; that is an implementation accident of
                   that arch, NOT a contract, so new families start at "copy"
                   and only move to "reassign" once
                   `caches.check_snapshot_semantics` has been run against them.

Families that ship an MTP head upstream but are NOT registered here, because
nothing in this repo can test them today:

  glm5_next   GLM-5.3 ships an MTP layer (layer 45, its own full expert
              stack — see families.py). mlx-lm has no glm5_next class yet, so
              there is no arch module to build a head against.
  deepseek_v3 DeepSeek's MTP module is a different shape again (its own
              embed/norm/head rather than a shared lm_head).

Registering either means writing its head module and running
`caches.check_snapshot_semantics` plus the acceptance probe first. A table
entry without a measured acceptance number is not evidence of anything.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class FamilySpec:
    name: str
    head: str
    capture: str
    draft_cache: str
    sidecar_name: str = "mtp-head-q6.safetensors"
    cache_semantics: str = "copy"

    def head_cls(self):
        mod, _, attr = self.head.partition(":")
        return getattr(importlib.import_module(mod), attr)

    def arch_module(self, model):
        """The module the trunk's classes were defined in. Artifacts ship a
        `model.py` that subclasses the registry arch, so walk to the core."""
        return importlib.import_module(type(model.model).__module__)

    def make_draft_cache(self, arch):
        try:
            return getattr(arch, self.draft_cache)()
        except AttributeError as e:
            raise RuntimeError(
                f"family {self.name}: architecture module {arch.__name__} has "
                f"no {self.draft_cache!r}; the registry entry is stale against "
                f"the installed mlx-lm") from e


FAMILIES: dict[str, FamilySpec] = {}


def register(spec: FamilySpec, *, replace: bool = False) -> FamilySpec:
    if spec.name in FAMILIES and not replace:
        raise ValueError(f"family already registered: {spec.name}")
    if spec.cache_semantics not in ("reassign", "copy"):
        raise ValueError(f"cache_semantics must be 'reassign' or 'copy', "
                         f"got {spec.cache_semantics!r}")
    FAMILIES[spec.name] = spec
    return spec


def unregister(name: str) -> None:
    FAMILIES.pop(name, None)


def model_type_of(model) -> str | None:
    """mlx-lm keeps the resolved config on `model.args`; multimodal configs
    nest the text half, and the top-level type is the one that names the
    architecture module."""
    for obj in (getattr(model, "args", None), model):
        mt = getattr(obj, "model_type", None)
        if isinstance(mt, str):
            return mt
    return None


def resolve(model, family: str | None = None) -> FamilySpec:
    if family is not None:
        if family not in FAMILIES:
            raise KeyError(f"unknown MTP family {family!r}; registered: "
                           f"{sorted(FAMILIES)}")
        return FAMILIES[family]
    mt = model_type_of(model)
    if mt in FAMILIES:
        return FAMILIES[mt]
    raise KeyError(
        f"no MTP family registered for model_type {mt!r}; registered: "
        f"{sorted(FAMILIES)}. Adding one is a FamilySpec in "
        f"vqlab/mtp/registry.py plus a head module — read that docstring.")


# ------------------------------------------------------------------ builtins
register(FamilySpec(
    name="qwen4_exp",
    head="vqlab.mtp_head:MTPHead",
    # The head drafts from the trunk activation that goes INTO the hyper-
    # connection mixer, i.e. the last thing before the final norm + lm_head.
    capture="hyper_connection_mixer",
    draft_cache="_AttnCache",
    sidecar_name="mtp-head-q6.safetensors",
    # Measured: every qwen4_exp cache slot is REASSIGNED (cache[0] = ...),
    # never mutated, and mlx arrays are immutable, so holding the old
    # references is a free snapshot. Verified by
    # caches.check_snapshot_semantics in tests/test_mtp_caches.py.
    cache_semantics="reassign",
))
