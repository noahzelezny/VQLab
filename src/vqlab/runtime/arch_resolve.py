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
# Layout no longer disqualifies an artifact: mlx_lm-layout module paths are
# remapped by _reach_vq (PLE `shard_N` -> `shards.N`) and mlx_lm-layout weight
# keys by the bundle's sanitize() on the mlx_vlm branch. F154's "cannot be
# served" verdict for Flash-Next was a naming mismatch, not a missing module.
_VISION_SERVABLE = bool(_MULTIMODAL)


def _loading_runtime():
    """Which runtime is importing this file, by inspecting the import stack.

    For several families ONE binding cannot serve both runtimes, so the
    bundle must follow its caller rather than pick once. qwen3_5 / qwen3_5_moe
    are the proof: mlx_vlm's arch builds its own cache types and its linear
    attention indexes them (`cache[0]`), while mlx_lm's loader hands it a bare
    KVCache -- so an mlx_vlm-bound bundle serves vision correctly and then
    dies on mlx_lm's text generation with `'KVCache' object is not
    subscriptable`. Binding once, either way, breaks one of the two.

    The loader that imports us is on the stack at import time, and that is the
    runtime whose conventions will be used for the rest of this model's life.
    sys.modules is NOT a usable signal: a process that merely imported mlx_vlm
    earlier would drag an mlx_lm load onto the wrong arch."""
    import inspect as _inspect
    import pathlib as _pl
    for _f in _inspect.stack():
        # PurePath.parts, not string surgery on separators -- the first cut of
        # this did a backslash replace and shipped a SyntaxError into the
        # bundle, which the MoE bundler wrote without compiling it.
        _parts = _pl.PurePath(_f.filename).parts
        if "mlx_vlm" in _parts:
            return "mlx_vlm"
        if "mlx_lm" in _parts:
            return "mlx_lm"
    return None


_LOADER = _loading_runtime()
if _VISION_SERVABLE and _LOADER == "mlx_lm":
    # Loaded for TEXT by mlx_lm: take its arch. Vision is unreachable on this
    # path and always was; text serving is what mlx_lm is being asked for.
    _arch = _resolve_arch(_cfg["model_type"], False)
else:
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
    import re as _re
    _cands = [_path]
    if _path.startswith("language_model."):
        _cands.append(_path[len("language_model."):])
    else:
        _cands.append("language_model." + _path)
    # PLE n-gram shards: mlx_lm registers them as attributes `shard_N`,
    # mlx_vlm keeps them in a list, `shards.N`. Same tensors, same order.
    # Flash-Next's config was written against mlx_lm and failed to attach
    # under mlx_vlm on exactly this (F154 called it "modules mlx_vlm does
    # not define"; it does define them, under the other spelling).
    _shard = _re.compile(r"\\.ngram_embedding\\.shard_(\\d+)$")
    for _c in list(_cands):
        if _shard.search(_c):
            _cands.append(_shard.sub(r".ngram_embedding.shards.\\1", _c))
    for _cand in _cands:
        _obj, _parts, _ok = _root, _cand.split("."), True
        for _c in _parts[:-1]:
            try:
                _obj = _obj[int(_c)] if _c.isdigit() else getattr(_obj, _c)
            except (AttributeError, IndexError, KeyError, TypeError):
                _ok = False
                break
        _leaf = _parts[-1]
        # The leaf may be a LIST INDEX, not an attribute: mlx_vlm keeps PLE
        # n-gram shards in a Python list (`shards.0`), and hasattr(list, "0")
        # is False. That single check is what kept Flash-Next's PLE modules
        # "unresolvable" under mlx_vlm after the shard_N -> shards.N remap.
        if _ok and _leaf.isdigit() and isinstance(_obj, (list, tuple)) \
                and int(_leaf) < len(_obj):
            return _obj, _leaf
        if _ok and hasattr(_obj, _leaf):
            return _obj, _leaf
    raise AttributeError(
        f"vq module {_path!r} does not resolve on this arch "
        f"({type(_root).__name__}); tried {_cands}. The bundle's config and "
        f"its base architecture disagree about the module tree.")


def _attach_vq(_owner, _leaf, _module):
    """Install `_module` at the leaf `_reach_vq` returned: index assignment
    when the owner is a list (mlx_vlm's PLE shards), setattr otherwise."""
    if _leaf.isdigit() and isinstance(_owner, list):
        _owner[int(_leaf)] = _module
    else:
        setattr(_owner, _leaf, _module)
'''

VLM_SANITIZE = '''
def _sanitize_for_vlm(_self, _weights):
    """Under the mlx_vlm arch, weight keys written in mlx_lm's layout
    (`model.layers.N...`, PLE `shard_N`) must be renamed onto mlx_vlm's tree
    or a strict load rejects them as unexpected. mlx_vlm's own sanitize_key
    only knows `model.language_model.*` and `model.visual.*`; it leaves bare
    `model.layers.*` alone. A no-op for artifacts already in mlx_vlm layout
    and never invoked on the mlx_lm branch."""
    import re as _re
    _shard = _re.compile(r"\\.ngram_embedding\\.shard_(\\d+)(?=\\.)")
    _out = {}
    for _k, _v in _weights.items():
        if _k.startswith("model.") and not _k.startswith(("model.visual.", "model.language_model.")):
            _k = "language_model." + _k
        _k = _shard.sub(r".ngram_embedding.shards.\\1", _k)
        _out[_k] = _v
    _base = getattr(super(type(_self), _self), "sanitize", None)
    return _base(_out) if _base is not None else _out
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
