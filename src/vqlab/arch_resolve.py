"""The base-runtime interop layer, as shim text shared by every bundler.

A bundle's `model.py` ships INSIDE the checkpoint and may not import vqlab, so
none of this can be a function the shim calls -- it is source text that each
bundler splices into the file it writes. It lives here because the MoE shim
(add_model_file) and the DENSE shim (dense_shim) each grew their own copy of
this interop and then DRIFTED: every fix landed on the MoE side only, and the
dense bundle was still missing the surface re-export and the config coercion
on 2026-09-19, three weeks after the MoE shim got them.

Three pieces, all about serving ONE bundle under EITHER runtime:

PRELUDE -- resolve the base arch and re-export its public surface.
    The bug this exists to prevent (2026-09-19): resolution was mlx_lm-first
    with an mlx_vlm fallback. That rule was written for glm5_next, which
    mlx_lm does NOT ship -- so the fallback always fired and the rule was
    never exercised against a model_type both runtimes define. Every other
    family is exactly that case: mlx_lm.models.{gemma4,qwen3_5,qwen3_5_moe,
    qwen4_exp} are TEXT-ONLY arches sharing a name with mlx_vlm's multimodal
    ones. A multimodal artifact bound to the text-only arch re-exports a
    surface with no TextConfig/VisionConfig/VisionModel, and mlx_vlm's loader
    dies in update_module_configs with `module 'custom_model' has no
    attribute 'TextConfig'` -- every vision tensor present on disk, the tower
    unreachable. 17 of 20 shipped multimodal bundles were in that state; only
    the three glm5_next ones escaped, and only by accident.
    The ARTIFACT decides, not whichever runtime happens to own the name: an
    artifact carrying a vision or audio tower prefers the multimodal arch.
    Text-only artifacts keep mlx_lm-first, unchanged.

COERCE -- `_coerce_module_configs(args)`, called first in Model.__init__.
    mlx_lm's loader passes nested module configs (text_config, vision_config,
    ...) through as plain DICTS; mlx_vlm's own loader coerces them to config
    objects first. A VLM arch then does `args.text_config.model_type` and dies
    on the dict. exo serves VLM artifacts through mlx_lm.utils.load_model,
    which is exactly the dict path. The hasattr(_arch, "TextConfig") guard
    scopes this to mlx_vlm-style arches: mlx_lm arches carry a nested `text`
    config their loader already parses natively, and update_module_configs
    against them dies on the missing TextConfig class (caught 2026-09-02 when
    a re-bundled Flash-Next 2.1bpw stopped loading under mlx_lm 0.31.9).

ARRAYISH -- `_arrayish(out)`, wrapped around Model.__call__'s return.
    mlx_lm's generate loop indexes the return value directly
    (`logits[:, -1, :]`); mlx_vlm callers read `.logits` and the other
    LanguageModelOutput dataclass fields. Text-only archs return a bare
    mx.array and pass straight through.
"""

PRELUDE = '''
def _resolve_arch(_mt, _multimodal):
    """Import the base arch for `_mt`, preferring the runtime that can serve
    this artifact's modalities. See vqlab/arch_resolve.py for why the order
    is conditional -- getting it backwards hides the vision tower."""
    _order = ("mlx_vlm", "mlx_lm") if _multimodal else ("mlx_lm", "mlx_vlm")
    _err = None
    for _pkg in _order:
        try:
            return _importlib.import_module(f"{_pkg}.models.{_mt}")
        except ModuleNotFoundError as _e:
            _err = _e
    raise _err


_MULTIMODAL = bool(_cfg.get("vision_config") or _cfg.get("audio_config"))

# Carrying a tower is necessary but NOT sufficient to serve it. The bundle can
# only use the multimodal arch if that arch hosts the modules this artifact's
# config names, and the two runtimes root the text model differently: mlx_lm
# at `model.layers.N`, mlx_vlm at `language_model.model.layers.N`. The config
# records which tree the artifact was BUILT against, so it -- not a guess --
# decides. 16 of the 20 multimodal artifacts here are mlx_vlm-shaped; the four
# Flash-Next rungs are mlx_lm-shaped AND carry PLE modules mlx_vlm's qwen4_exp
# does not define at all, under either spelling. Forcing them onto the
# multimodal arch does not reveal their tower, it stops them loading entirely
# -- text included. Serving text is strictly better than serving nothing, so
# they stay on mlx_lm and their vision stays unreachable until the artifact
# itself is rebuilt against the VLM module tree. `vision-smoke` reports that
# state as a FAIL rather than hiding it.
_VQ_KEYS = (list(_cfg.get("vq_modules", {}))
            + list(_cfg.get("vq_linear", {}))
            + list(_cfg.get("vq_embed", {}))
            + list((_cfg.get("vq_ple") or {}).get("keys", [])))
_VLM_LAYOUT = any(_k.startswith("language_model.") for _k in _VQ_KEYS)
_VISION_SERVABLE = bool(_MULTIMODAL and (_VLM_LAYOUT or not _VQ_KEYS))
_arch = _resolve_arch(_cfg["model_type"], _VISION_SERVABLE)

# Re-export the base module's WHOLE public surface, not a hand-listed few.
# A VLM base is read for far more than Model: mlx_vlm's loader reaches for
# TextConfig, VisionConfig, VisionModel, LanguageModel... Enumerating them one
# at a time is whack-a-mole against a surface we do not own, and each miss is
# an AttributeError at load. Model is excluded: our subclass below replaces it.
for _n in dir(_arch):
    if not _n.startswith("_") and _n != "Model":
        globals()[_n] = getattr(_arch, _n)

# The two runtimes read DIFFERENT names for the args class: mlx_lm wants
# ModelArgs, mlx_vlm calls model_class.ModelConfig.from_dict(config). Export
# whichever the base defines under both names so one bundle serves either.
ModelArgs = getattr(_arch, "ModelArgs", None) or _arch.ModelConfig
ModelConfig = getattr(_arch, "ModelConfig", None) or _arch.ModelArgs
'''

COERCE = '''
def _coerce_module_configs(_args):
    """Turn mlx_lm's plain-dict nested configs into the config objects an
    mlx_vlm arch expects. No-op for text-only models and for mlx_vlm-loaded
    ones. See vqlab/arch_resolve.py."""
    if not (isinstance(getattr(_args, "text_config", None), dict)
            and hasattr(_arch, "TextConfig")):
        return _args
    from mlx_vlm.utils import (
        apply_generation_config_defaults,
        update_module_configs,
    )
    _args = update_module_configs(
        _args, _arch, _cfg,
        ["text", "vision", "perceiver", "projector", "audio"])
    return apply_generation_config_defaults(_args, _cfg)
'''

PATHWALK = '''
def _reach_vq(_root, _path):
    """Walk `_path` to (owner, leaf name), tolerating the two module layouts.

    config's vq_module paths are written in whatever layout the artifact was
    BUILT against: the mlx_lm arch roots the text model at `model.layers.N`,
    while the mlx_vlm arch nests it at `language_model.model.layers.N`. The
    fleet contains both spellings -- gemma was built mlx_vlm-style, Flash-Next
    mlx_lm-style -- so once arch resolution started following the artifact's
    modalities (see above), Flash-Next's paths stopped resolving and the
    bundle failed to load AT ALL, text included. Rebasing here keeps the
    published config.json untouched, which matters: the config is the record
    of what shipped and a path rewrite would silently invalidate it.

    Tries the path as written, then with `language_model.` added, then with it
    removed. Raises naming every candidate -- a VQ module that silently fails
    to attach leaves a dense random-init layer in the graph, which loads
    clean and generates plausible garbage."""
    _cands = [_path]
    if _path.startswith("language_model."):
        _cands.append(_path[len("language_model."):])
    else:
        _cands.append("language_model." + _path)
    for _cand in _cands:
        _obj, _parts, _ok = _root, _cand.split("."), True
        for _c in _parts[:-1]:
            try:
                _obj = _obj[int(_c)] if _c.isdigit() else getattr(_obj, _c)
            except (AttributeError, IndexError, KeyError, TypeError):
                _ok = False
                break
        if _ok and hasattr(_obj, _parts[-1]):
            return _obj, _parts[-1]
    raise AttributeError(
        f"vq module {_path!r} does not resolve on this arch "
        f"({type(_root).__name__}); tried {_cands}. The bundle's config and "
        f"its base architecture disagree about the module tree.")
'''

ARRAYISH = '''
_ARRAYISH_CACHE = {}


def _arrayish(_out):
    """Let one return value satisfy both runtimes: mlx_vlm's dataclass fields
    AND mlx_lm's direct indexing. See vqlab/arch_resolve.py."""
    if isinstance(_out, mx.array) or not hasattr(_out, "logits"):
        return _out
    _cls = type(_out)
    _sub = _ARRAYISH_CACHE.get(_cls)
    if _sub is None:
        _sub = type(_cls.__name__, (_cls,), {
            "__getitem__": lambda _s, _k: _s.logits[_k],
            "shape": property(lambda _s: _s.logits.shape),
            "dtype": property(lambda _s: _s.logits.dtype),
            "ndim": property(lambda _s: _s.logits.ndim),
        })
        _ARRAYISH_CACHE[_cls] = _sub
    _out.__class__ = _sub
    return _out
'''

# Back-compat alias: the first cut of this module exported only the resolver.
SNIPPET = PRELUDE
