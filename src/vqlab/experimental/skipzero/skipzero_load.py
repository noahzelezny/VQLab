"""vq-skipzero (EXPERIMENTAL) load-time expansion shim.

This file is copied VERBATIM into a vq-skipzero artifact next to its model.py.
The generated model.py wraps the bundle's Model with sanitize()/load_weights()
hooks that call expand_weights() here, so the UNCHANGED vq_switch runtime
receives ordinary full [E, OUT, W] codes / vq_scales tensors.

On-disk form, per VQ switch module prefix {p} (replaces {p}.codes and
{p}.vq_scales; {p}.codebook is untouched):
    {p}.sz_shape    int32 [4]              E, OUT, code words per row, scales per row
    {p}.sz_rowmask  uint8 [E, ceil(OUT/8)] bit-packed LIVE-row mask, little bit order
    {p}.sz_codes    codes dtype [NLIVE, W] live rows only, row-major over (e, out)
    {p}.sz_scales   scales dtype [NLIVE, G]
Dead rows (every scale of the row < 6.2e-5 in the source) expand to code 0 and
scale 0, i.e. EXACT zero weights. Live rows are byte-identical to the source.

Stage 1 saves DISK only: after expansion the in-memory tensors are the same
size as the unpacked artifact's.
"""
from __future__ import annotations

SUFFIXES = ("sz_shape", "sz_rowmask", "sz_codes", "sz_scales")
FORMAT = "vq-skipzero"
VERSION = 1
# Codebook dims the runtime row-table switch can SERVE compact. vq_switch.py's
# gemmseg prefill is the binding site: it raises "SKIPZERO prefill: gemmseg2
# d2/d4/d8 with VQ_SPEC_KERNELS=1 only" for any other dim. d2 and d8 are
# switched only in the PACKED decode kernels (d2 walk; d8 walk and simd
# devx_ss), so an unpacked d2 or d8 module is not servable. sz-pack packs and
# check-bundle gates against servable().
SUPPORTED_DIMS = (2, 4, 8)
PACKED_ONLY_DIMS = (2, 8)


def servable(spec):
    """Can the runtime serve this vq_modules entry with a row table?"""
    d = (spec or {}).get("dim")
    return d in SUPPORTED_DIMS and (d not in PACKED_ONLY_DIMS
                                    or bool((spec or {}).get("pack_bits")))


def _prefixes(keys):
    return sorted({k[: -len(".sz_rowmask")] for k in keys if k.endswith(".sz_rowmask")})


# ------------------------------------------------------------------- numpy
def live_mask_np(rowmask, E, OUT):
    import numpy as np
    bits = np.unpackbits(np.asarray(rowmask, np.uint8), axis=-1, bitorder="little")
    return bits[:, :OUT].astype(bool).reshape(E, OUT)


def expand_np(shape, rowmask, compact):
    """compact [NLIVE, W] -> full [E, OUT, W]; dead rows are all-zero bytes."""
    import numpy as np
    E, OUT = int(shape[0]), int(shape[1])
    live = live_mask_np(rowmask, E, OUT).reshape(-1)
    compact = np.asarray(compact)
    if compact.shape[0] != int(live.sum()):
        raise ValueError(f"skipzero: {compact.shape[0]} compact rows vs {int(live.sum())} live bits")
    full = np.zeros((E * OUT,) + compact.shape[1:], compact.dtype)
    full[live] = compact
    return full.reshape((E, OUT) + compact.shape[1:])


# --------------------------------------------------------------------- mlx
def expand_mx(shape, rowmask, compact):
    """Same as expand_np, lazily in MLX (a gather + where; no nonzero needed)."""
    import mlx.core as mx
    shape = [int(v) for v in (shape.tolist() if hasattr(shape, "tolist") else shape)]
    E, OUT = shape[0], shape[1]
    W = compact.shape[1]
    if compact.shape[0] == 0:
        return mx.zeros((E, OUT, W), dtype=compact.dtype)
    rm = rowmask.astype(mx.uint32)
    bits = (rm[..., None] >> mx.arange(8, dtype=mx.uint32)) & 1        # [E, B, 8]
    live = bits.reshape(E, -1)[:, :OUT].reshape(-1)                     # [E*OUT]
    pos = mx.maximum(mx.cumsum(live.astype(mx.int32)) - 1, 0)
    rows = mx.take(compact, pos, axis=0)                                # [E*OUT, W]
    full = mx.where(live[:, None] > 0, rows, mx.zeros((1, W), dtype=compact.dtype))
    return full.reshape(E, OUT, W)


def expand_weights(weights, backend="mx"):
    """Replace every skipzero quadruple with ordinary {p}.codes / {p}.vq_scales.

    Accepts a dict or a list of (key, array) pairs and returns the same kind.
    Idempotent: weights without sz_ keys pass through untouched, so both the
    sanitize() and the load_weights() hook may call it."""
    as_list = not isinstance(weights, dict)
    d = dict(weights) if as_list else weights
    pre = _prefixes(d)
    if not pre:
        return weights
    ex = expand_mx if backend == "mx" else expand_np
    out = _drop(d, pre)
    for p in pre:
        shp, rmask = d[p + ".sz_shape"], d[p + ".sz_rowmask"]
        out[p + ".codes"] = ex(shp, rmask, d[p + ".sz_codes"])
        out[p + ".vq_scales"] = ex(shp, rmask, d[p + ".sz_scales"])
    return list(out.items()) if as_list else out


def _drop(d, pre):
    kill = {p + "." + s for p in pre for s in SUFFIXES}
    return {k: v for k, v in d.items() if k not in kill}


# text appended to the bundle's model.py by sz-pack (kept here so the loader
# hook and the expansion it calls are versioned together)
MODEL_HOOK = '''

# ===========================================================================
# vq-skipzero (EXPERIMENTAL, not a shipped format) -- appended by
# `vqlab sz-pack`. Everything ABOVE this line is the source bundle's model.py,
# byte-for-byte. The hook expands the compact dead-row-free tensors back to
# full [E, OUT, W] codes / vq_scales (dead rows = code 0, scale 0) before the
# unchanged VQSwitchLinear sees them. BOTH loaders are covered: mlx_lm calls
# sanitize(); mlx_vlm skips sanitize for format=mlx shards but always calls
# load_weights(). expand_weights() is idempotent, so running twice is safe.
# ===========================================================================
import importlib.util as _sz_util
_sz_spec = _sz_util.spec_from_file_location(
    "skipzero_load", _pathlib.Path(__file__).parent / "skipzero_load.py")
_skipzero = _sz_util.module_from_spec(_sz_spec)
_sz_spec.loader.exec_module(_skipzero)

_SZBaseModel = Model


class Model(_SZBaseModel):
    def sanitize(self, weights):
        weights = _skipzero.expand_weights(weights)
        _base = getattr(super(), "sanitize", None)
        return _base(weights) if _base is not None else weights

    def load_weights(self, file_or_weights, strict=True):
        if not isinstance(file_or_weights, (str, _pathlib.Path)):
            file_or_weights = _skipzero.expand_weights(file_or_weights)
        return super().load_weights(file_or_weights, strict=strict)
'''
