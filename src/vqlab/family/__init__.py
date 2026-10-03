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
        reference="org/repo: inference/model.py",   # the OFFICIAL inference code
        parity=(Parity(...), ...),          # reference behaviours + their tests
        fused=((a, b, fused), ...),         # projections the runtime fuses
        act_stats=fn,                       # how often PARITY's clamps fire
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
class Parity:
    """One behaviour of the family's REFERENCE inference code that VQLab's
    runtime must reproduce (F195: DeepSeek's reference clamps the shared
    expert's SwiGLU; mlx-lm did not, and every KL number used the wrong one).

    test: "tests/<file>.py::<function>" that checks it, or "" = UNTESTED
    (say so; never point at a test that does not check the behaviour).
    required: `vqlab parity` exits non-zero while a required item is untested.
    probe: the act-stats counter that measures how often it fires, if any."""
    name: str
    reference: str              # what the reference does (file:line where useful)
    runtime: str                # where/how VQLab / mlx-lm matches it (or doesn't)
    test: str = ""
    required: bool = True
    probe: str = ""


@dataclasses.dataclass(frozen=True)
class FamilyPlugin:
    name: str
    model_types: tuple = ()
    fit: Optional[dict] = None
    scorer: Optional[dict] = None
    tokenizer_register: Optional[Callable[[str], None]] = None
    notes: str = ""
    reference: str = ""
    parity: tuple = ()
    # (a, b, fused) module-name suffixes: the runtime's sanitize fuses two
    # projections that share their input; checkpoints store them apart.
    fused: tuple = ()
    # act_stats(model_dir, tokens, corpora, args) -> iterable of result dicts
    act_stats: Optional[Callable] = None


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


def get(name: str) -> Optional[FamilyPlugin]:
    """The plugin serving a family name or model_type, else None."""
    for p in plugins().values():
        if name == p.name or name in p.model_types:
            return p
    return None


def fused_projections(name: str) -> tuple:
    """(a, b, fused) suffix triples the family's runtime fuses; () if none."""
    p = get(name)
    return tuple(p.fused) if p else ()


def alias_fused(skel: dict, triples) -> dict:
    """Give each fused module the quantization of its two halves, in place.

    skel maps module path -> {"bits", "group_size", ...}. For every (a, b,
    fused) triple whose halves BOTH appear under one prefix, skel[prefix +
    fused] = skel[prefix + a]. Affine groups run along the INPUT dimension,
    so quantizing the fused matrix at the halves' common width is the same
    as quantizing each half; differing widths raise ValueError."""
    for k in list(skel):
        for x, y, fused in triples:
            if k.endswith("." + x):
                pre = k[: -len(x)]
                if pre + y not in skel:
                    continue
                a_, b_ = skel[k], skel[pre + y]
                if (a_.get("bits"), a_.get("group_size")) != (b_.get("bits"), b_.get("group_size")):
                    raise ValueError(f"{k} and {pre + y} have different widths; cannot fuse into {fused}")
                skel[pre + fused] = a_
    return skel


def variant_for(model_type: str):
    """The scorer's current numerics variant for model_type (None when the
    family declares none). Stamped into every cache and result by
    core.numerics.build."""
    fn = (scorers().get(model_type) or {}).get("variant")
    return fn() if callable(fn) else None
