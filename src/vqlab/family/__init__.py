"""Model families as plugins: one file per family (structural feedback #2).

Onboarding DeepSeek-V4-Flash touched six places: core/families.py (source
layout), core/expert_src.py (FP4 reader), score/stream_score.py (scorer),
gate/check_release.py (tokenizer registration), the MTP head scripts and a
fitter guard. A plugin module here declares a family's pieces in one place:

    SPEC = FamilyPlugin(
        name="deepseek_v4",                 # FAMILY key (fit-moe --family, verify)
        model_types=("deepseek_v4",),       # checkpoint model_type(s) it serves
        fit={...},                          # the FAMILY entry: source layout, projections
        scorer={"fn": score, "family": "deepseek_v4", "validated": True, ...},
        tokenizer_register=fn,              # make AutoTokenizer load (optional)
    )

and the shared tables read it: core.families.FAMILY, stream_score.SCORERS,
check-release's tokenizer probe. Every module in this package is a plugin;
nothing else needs editing for a new family. Families that predate this
package still live in those tables directly and move here one at a time.
"""
from __future__ import annotations

import dataclasses
import importlib
import pathlib
from typing import Callable, Optional


@dataclasses.dataclass(frozen=True)
class FamilyPlugin:
    name: str
    model_types: tuple = ()
    fit: Optional[dict] = None
    scorer: Optional[dict] = None
    tokenizer_register: Optional[Callable[[str], None]] = None
    notes: str = ""


_CACHE: dict = {}


def plugins() -> dict:
    """{name: FamilyPlugin} for every module in this package."""
    if not _CACHE:
        for f in sorted(pathlib.Path(__file__).parent.glob("*.py")):
            if f.name.startswith("_"):
                continue
            mod = importlib.import_module(f"vqlab.family.{f.stem}")
            spec = getattr(mod, "SPEC", None)
            if isinstance(spec, FamilyPlugin):
                _CACHE[spec.name] = spec
    return _CACHE


def fit_families() -> dict:
    return {p.name: p.fit for p in plugins().values() if p.fit}


def scorers() -> dict:
    """{model_type: scorer entry} (stream_score.SCORERS shape)."""
    out = {}
    for p in plugins().values():
        if p.scorer:
            for mt in p.model_types or (p.name,):
                out[mt] = p.scorer
    return out


def tokenizer_hook(model_type: str) -> Optional[Callable[[str], None]]:
    for p in plugins().values():
        if model_type in (p.model_types or (p.name,)) and p.tokenizer_register:
            return p.tokenizer_register
    return None


def variant_for(model_type: str):
    """The scorer's current numerics variant for model_type (None when the
    family declares none). Stamped into every cache and result by
    core.numerics.build."""
    fn = (scorers().get(model_type) or {}).get("variant")
    return fn() if callable(fn) else None
