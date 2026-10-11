"""The sz-bitexact gate's CPU self-check (was skipzero/sz_bitexact.py
`selftest`, moved here in untangle step 14). No GPU: the row table, the
resident-weights contract, and that the kernel forks apply to each runtime
vintage's text. Kernel execution needs the GPU: `vqlab sz-bitexact
--synthetic` and the model-mode gate.

    vqlab sz-bitexact --selftest
    python -m vqlab.selftest.sz_bitexact_cpu
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

from vqlab.skipzero import skipzero_load, sz_resident
from vqlab.skipzero.sz_bitexact import SEED, SRC, U8_MODEL_PY, WALK_FROZEN


def _src_strings(path, names):
    """Kernel source strings of a runtime/model.py FILE, by AST (no import)."""
    return sz_resident.kernel_srcs_from_text(pathlib.Path(path).read_text(), tuple(names))


def _check_row_table_np(E, OUT, G, live, rowmask, tbl, comp):
    """(1) the row table addresses exactly the rows the stage-1 expansion places"""
    full = skipzero_load.expand_np([E, OUT, comp.shape[1], G], rowmask, comp)
    g = np.where(tbl[..., None] >= 0, comp[np.maximum(tbl, 0)], 0).astype(np.uint32)
    t1 = g.tobytes() == full.tobytes() and (tbl < 0).sum() == (~live).sum()
    print(f"[selftest] row table vs stage-1 expansion (numpy): {'OK' if t1 else 'FAIL'}")
    return t1


def _check_row_table_mx(E, OUT, W, G, rowmask, tbl, comp):
    """(2) mlx-CPU row table == numpy row table; resident_weights shape contract"""
    try:
        import mlx.core as mx
        with mx.stream(mx.cpu):
            tm = sz_resident.row_table_mx(mx.array(np.array([E, OUT, W, G], np.int32)),
                                          mx.array(rowmask))
            w = sz_resident.resident_weights({
                "a.sz_shape": mx.array(np.array([E, OUT, W, G], np.int32)),
                "a.sz_rowmask": mx.array(rowmask), "a.sz_codes": mx.array(comp),
                "a.sz_scales": mx.array(np.zeros((comp.shape[0], G), np.float16)),
                "b.weight": mx.array(np.ones(3, np.float16))})
            mx.eval(tm, *w.values())
            t2 = (np.array(tm) == tbl).all() and set(w) == {"a.codes", "a.vq_scales",
                                                            "a.row_table", "b.weight"} \
                and w["a.codes"].shape == comp.shape and w["a.row_table"].dtype == mx.int32
            t2 &= sz_resident.resident_weights(w) is w          # idempotent
        print(f"[selftest] mlx-CPU row table + resident_weights: {'OK' if t2 else 'FAIL'}")
        return bool(t2)
    except ImportError:
        print("[selftest] mlx not importable: SKIPPED the mlx-CPU row-table check")
        return True


def _check_native_switch():
    # Since 2026-09-29 runtime/vq_switch.py carries the SZ switch NATIVELY
    # (`#if SZ` in the walk decode kernel and gemmseg2), so the walk fork now
    # targets the last runtime WITHOUT it -- the text every walk-vintage
    # bundle built before then carries (WALK_FROZEN).
    rt_txt = (SRC / "vqlab/runtime/vq_switch.py").read_text()
    native = (rt_txt.count("#if SZ") >= 2 and "def skipzero_weights" in rt_txt
              and "row_table=None" in rt_txt)
    print(f"[selftest] runtime/vq_switch.py carries the SZ switch natively: "
          f"{'OK' if native else 'FAIL'}")
    return native


def _forks_apply(s, want):
    """The vintage is detected from the text and the OTHER vintage's patch
    set must refuse it. Returns (ok, detected vintage)."""
    try:
        v, d, p = sz_resident.patch_vintage(s)
        t = (v == want and d.count("rowtbl") == 1 and p.count("rowtbl") == 1
             and "sz_lr < 0" in d)
        if want == "u8":
            t &= "#if BITS != 0 || D_BAKE != 4" in p and "(size_t)e * OUT * WPR" not in p
            t &= "srow_w = scales\n" not in p
            try:           # the walk gemmseg patch must REFUSE this text
                sz_resident.patch_gemmseg_src(s["_SRC_GEMMSEG2"])
                t = False
            except RuntimeError:
                pass
        else:
            try:
                sz_resident.patch_gemmseg_u8_src(s["_SRC_GEMMSEG2"])
                t = False
            except RuntimeError:
                pass
    except (RuntimeError, NotImplementedError) as e:
        print(e)
        t, v = False, None
    return t, v


def _check_vintage_forks():
    """(3)/(4) kernel forks apply to each vintage's text (no import, no GPU)."""
    ok = True
    names = ["_SRC_FUSED_PACKED", "_SRC_FUSED_PACKED_D4_WALK", "_SRC_GEMMSEG2"]
    for label, path, want in (("walk runtime (frozen)", WALK_FROZEN, "walk"),
                              ("397B bundle model.py", U8_MODEL_PY, "u8")):
        if path is None:
            print(f"[selftest] {label}: none named (set VQLAB_SZ_U8_MODEL_PY): SKIPPED")
            continue
        path = pathlib.Path(path)
        if not path.exists():
            if want == "u8":   # a local artifact, not package data
                print(f"[selftest] {label}: {path} not on this machine: SKIPPED")
                continue
            print(f"[selftest] {label}: {path} not found: FAIL")
            ok = False
            continue
        if want == "u8":
            txt = path.read_text()
            if txt.count("#if SZ") >= 2 and "def skipzero_weights" in txt:
                # rebundled onto a runtime with the SZ switch built in (the
                # 397B was, 2026-10-01): no fork applies, none is needed
                print(f"[selftest] {label}: carries the SZ switch natively (rebundled), "
                      "no u8 fork needed: OK")
                continue
        t, v = _forks_apply(_src_strings(path, names), want)
        print(f"[selftest] vintage '{v}' detected + forks apply to {label}: {'OK' if t else 'FAIL'}")
        ok &= bool(t)
    return ok


def _check_u8_rows(rng, E, OUT, G, live, rowmask, tbl):
    """(5) u8 compact rows through the row table == stage-1 expansion (numpy)"""
    comp8 = rng.integers(0, 256, (int(live.sum()), 8)).astype(np.uint8)
    full8 = skipzero_load.expand_np([E, OUT, 8, G], rowmask, comp8)
    g8 = np.where(tbl[..., None] >= 0, comp8[np.maximum(tbl, 0)], 0).astype(np.uint8)
    t5 = g8.tobytes() == full8.tobytes() and (comp8.view(np.uint32).view(np.uint8) == comp8).all()
    print(f"[selftest] u8 compact rows via row table vs stage-1 expansion: {'OK' if t5 else 'FAIL'}")
    return bool(t5)


def run():
    ok = True
    rng = np.random.default_rng(SEED)
    E, OUT, W, G = 6, 77, 5, 3
    live = rng.random((E, OUT)) > 0.7
    live[0] = False
    rowmask = np.packbits(live, axis=-1, bitorder="little")
    tbl = sz_resident.row_table_np(rowmask, E, OUT)
    comp = rng.integers(0, 2**32, (int(live.sum()), W), dtype=np.uint64).astype(np.uint32)
    ok &= _check_row_table_np(E, OUT, G, live, rowmask, tbl, comp)
    ok &= _check_row_table_mx(E, OUT, W, G, rowmask, tbl, comp)
    ok &= _check_native_switch()
    ok &= _check_vintage_forks()
    ok &= _check_u8_rows(rng, E, OUT, G, live, rowmask, tbl)
    print("[selftest] SKIPPED (need the GPU): kernel execution -- run "
          "`vqlab sz-bitexact --synthetic` and the model-mode gate")
    return 0 if ok else 1


def main(argv=None):
    return run()


if __name__ == "__main__":
    sys.exit(main())
