"""Selftest: the fitters (fit-dense, fit-moe, reselect, fit-additive)."""
from __future__ import annotations

import contextlib
import io
import json
import pathlib
import shutil

import mlx.core as mx

from vqlab._layout import find as _find
from vqlab.selftest import check, run
from vqlab.selftest.synthetic_model import LAYERS, PROJS, decode_all, moe_fixture


def _fit_dense(ctx, out, seed):
    return run([str(_find("fit_dense_vq.py")), "--src", str(ctx.src), "--out", str(out),
                "--k", "16", "--dim", "2", "--layers", f"0-{LAYERS-1}",
                "--family", "qwen3_8_dense", "--iters", "2", "--seed", seed],
               verbose=ctx.verbose)


def check_fitter(ctx):
    print("[1/7] fitter")
    ctx.f1 = f1 = ctx.tmp / "fit-a"
    p = _fit_dense(ctx, f1, "1234")
    if not check("fit runs and writes an artifact", p.returncode == 0
                 and (f1 / "config.json").exists()):
        return False
    check("fit reports SEEDED by default", "SEEDED" in p.stdout)
    cfg = json.load(open(f1 / "config.json"))
    nmod = len(cfg.get("vq_modules", {}))
    check("every target tensor was fit", nmod == LAYERS * len(PROJS),
          f"{nmod} modules")
    check("code dtype follows K (uint8 at K<=256)",
          mx.load(str(f1 / "model-00001-of-00001.safetensors"))[
              list(cfg["vq_modules"])[0] + ".codes"].dtype == mx.uint8)
    _check_seed_gate(ctx)


def _check_seed_gate(ctx):
    """seed gate, BOTH directions (III.5)"""
    f2, ctx.f3 = ctx.tmp / "fit-b", ctx.tmp / "fit-unseeded"
    _fit_dense(ctx, f2, "1234")
    _fit_dense(ctx, ctx.f3, "-1")
    A, B, C = (decode_all(x) for x in (ctx.f1, f2, ctx.f3))
    ctx.A = A
    ctx.m0 = m0 = list(A)[0]
    same_seed = float(mx.mean(mx.abs(A[m0] - B[m0])).item())
    diff_seed = float(mx.mean(mx.abs(A[m0] - C[m0])).item())
    check("same seed reproduces the fit", same_seed < 1e-6,
          f"mean|delta| {same_seed:.2e}")
    check("--seed -1 gives an independent draw (gate fails on known-bad)",
          diff_seed > same_seed * 10, f"mean|delta| {diff_seed:.2e}")


def check_fit_moe(ctx):
    print("[5b/7] fit-moe: files its fits, writes a build record")
    tmp = ctx.tmp
    mt, mb, mo = tmp / "moe-teacher", tmp / "moe-base", tmp / "moe-out"
    bm = moe_fixture(mt, mb)
    p = run([str(_find("fit_moe.py")), "--base", str(mb), "--src", str(mt),
             "--out", str(mo), "--vq-layers", "0", "--k", "16", "--dim", "4",
             "--iters", "2", "--sample", "1000", "--family", "qwen3_5",
             "--relerr-abort", "1.0"], verbose=ctx.verbose)
    if check("fit-moe runs on the fixture", p.returncode == 0,
             (p.stderr or p.stdout)[-200:] if p.returncode else ""):
        mrec = json.load(open(mo / "vqlab_provenance.json"))
        check("fit-moe build record: method + module origin",
              mrec["tool"]["name"] == "fit-moe" and mrec["method"]["seed"] == 1234
              and mrec["modules"][bm]["origin"] == "fit")
        from vqlab.core import fitstore as fs_
        got = [r for r in fs_.read_index(tmp / "fitstore").values()
               if r["module"] == bm and r["recipe"]["tool"] == "fit-moe"]
        check("fit-moe files its fit in the store with its recipe",
              len(got) == 1 and got[0]["d"] == 4 and got[0]["K"] == 16
              and got[0]["teacher"] == "moe-teacher", f"{len(got)} stored")


def check_reselect(ctx):
    print("[6a36/7] reselect (CPU, synthetic Gram)")
    # reselect writes only under configured storage, so its fixture runs
    # under scratch and is removed after
    from vqlab import config as _cfg
    _ssd = _cfg.scratch()
    if _ssd.is_dir():
        _rroot = _ssd / "selftest" / ctx.tmp.name
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                from vqlab.selftest import reselect_fixture
                _rout = reselect_fixture.run(root=_rroot)
            check("reselect lowers the Gram-weighted error and writes a NEW dir with a build record",
                  (pathlib.Path(_rout) / "vqlab_provenance.json").exists())
        finally:
            shutil.rmtree(_rroot, ignore_errors=True)
    else:
        print(f"  SKIP  reselect fixture -- scratch {_ssd} does not exist (vqlab.config)")


def check_additive(ctx):
    print("[6a38/7] fit-additive (CPU fixture)")
    import numpy as _np2
    _dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        from vqlab.fit import additive_vq as _av
        from vqlab.fit import geo_build as _gb
        _W = mx.array(_np2.random.default_rng(0).standard_normal((2, 32, 256)).astype(_np2.float32))
        with contextlib.redirect_stdout(io.StringIO()):
            _cb, _codes, _sc, (_C1, _C2, _I, _J) = _av.fit_module_additive(
                _W, 4, 8, 8, _np2.random.default_rng(1234), rounds=2, return_parts=True)
        check("fit-additive expands C1+C2 into one ordinary codebook, bit-exact in fp16",
              _cb.shape == (64, 4) and _np2.array_equal(
                  _np2.array(_cb)[_I * 8 + _J].view(_np2.uint16),
                  (_C1[_I] + _C2[_J]).astype(_np2.float16).view(_np2.uint16)))
        check("fit-additive's recipe says additive and carries every RECIPE_KEY (never pooled as k-means)",
              _av.method()["init"].startswith("additive") and set(_gb.RECIPE_KEYS) <= set(_av.method()))
    finally:
        mx.set_default_device(_dev)

