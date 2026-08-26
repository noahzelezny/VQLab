#!/usr/bin/env python3
"""vqlab selftest — run the real pipeline on a tiny synthetic model.

This is not a mock. It synthesizes a small checkpoint, then runs the SHIPPED
fitter, gate, packer, manifest and kernels over it as subprocesses, exactly
as a user would, and checks the properties each stage is supposed to
guarantee. It needs no downloaded model and finishes in well under a minute.

Every check that can be gated in both directions is (III.5: a gate must FAIL
on a known-bad input and PASS on a known-good one before its pass means
anything). Checks that would need a multi-GB real model — end-to-end
generation through mlx-lm, scoring — are reported as SKIPPED with the reason,
never silently omitted.

    vqlab selftest [--keep] [--verbose]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

import mlx.core as mx

HERE = pathlib.Path(__file__).parent
PY = sys.executable

# tiny geometry: IN divisible by group(64) and dim; OUT small.
LAYERS, OUT_D, IN_D, G = 2, 64, 128, 64
KEY = "model.language_model.layers.{li}.mlp.{key}.weight"
PROJS = ("gate_proj", "up_proj", "down_proj")

PASS, FAIL, SKIP = [], [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
    return ok


def skip(name, why):
    SKIP.append(name)
    print(f"  SKIP  {name} — {why}")


def run(args, expect_rc=0, verbose=False):
    p = subprocess.run([PY, *args], capture_output=True, text=True)
    if verbose or (p.returncode != expect_rc):
        print(p.stdout[-2000:], p.stderr[-2000:], sep="\n")
    return p


def make_source(d: pathlib.Path):
    """A tiny checkpoint shaped like a dense qwen MLP stack."""
    mx.random.seed(7)
    w, wm = {}, {}
    # Draw subvectors from a small set of centres so the tensors are actually
    # VQ-compressible, the way real weights are. Pure gaussian noise fits at
    # relerr ~0.34 even when healthy, which would leave a COLLAPSED tensor
    # (relerr 1.0) sitting below 3x the median — invisible to the relative
    # gate. A fixture has to be realistic enough for the gate under test to
    # be able to fire.
    centres = mx.random.normal((16, 2)) * 0.05
    for li in range(LAYERS):
        for proj in PROJS:
            k = KEY.format(li=li, key=proj)
            pick = mx.random.randint(0, centres.shape[0], (OUT_D * IN_D // 2,))
            sub = centres[pick] + mx.random.normal((OUT_D * IN_D // 2, 2)) * 0.002
            w[k] = sub.reshape(OUT_D, IN_D).astype(mx.bfloat16)
            wm[k] = "model-00001-of-00001.safetensors"
    d.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(d / "model-00001-of-00001.safetensors"), w)
    json.dump({"metadata": {}, "weight_map": wm},
              open(d / "model.safetensors.index.json", "w"), indent=1)
    json.dump({"model_type": "qwen3", "quantization": {"group_size": G, "bits": 4}},
              open(d / "config.json", "w"), indent=1)


def decode_all(art: pathlib.Path):
    """Decode every VQ tensor in an artifact to dense weights."""
    sys.path.insert(0, str(HERE))
    import vq_pack
    cfg = json.load(open(art / "config.json"))
    mods = cfg.get("vq_modules") or cfg.get("vq_linear") or {}
    idx = json.load(open(art / "model.safetensors.index.json"))["weight_map"]
    out = {}
    for m, meta in mods.items():
        shard = art / idx[m + ".codes"]
        with mx.stream(mx.cpu):
            data = mx.load(str(shard))
            mx.eval(list(data.values()))
        codes, cb = data[m + ".codes"], data[m + ".codebook"]
        sc = data[m + ".vq_scales"]
        D, K = int(cb.shape[1]), int(cb.shape[0])
        nsub = meta["in"] // D
        c = codes.reshape(-1, codes.shape[-1])
        if meta.get("pack_bits"):
            c = mx.array(vq_pack.unpack(np_of(c), nsub, meta["pack_bits"]))
        c = c.reshape(-1, nsub)
        w = cb.astype(mx.float32)[c.reshape(-1).astype(mx.int32)]
        w = w.reshape(c.shape[0], nsub * D)
        w = w.reshape(c.shape[0], meta["in"] // meta["group"], meta["group"])
        w = w * sc.reshape(-1, meta["in"] // meta["group"])[..., None].astype(mx.float32)
        out[m] = w.reshape(c.shape[0], meta["in"])
        mx.eval(out[m])
    return out


def np_of(a):
    import numpy as np
    return np.array(a)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vqlab selftest",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("--keep", action="store_true", help="keep the temp workspace")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="vqlab_selftest_"))
    v = a.verbose
    try:
        print("NOTE: this runs REAL Metal kernels and real k-means fits. It is "
              "small (seconds of GPU) but it CONTENDS.\n      Do not run it on a "
              "box that is mid-experiment — strictly sequential is the rule "
              "this project's\n      results depend on.\n")
        print(f"workspace: {tmp}\n")
        src = tmp / "src"
        make_source(src)

        # ---------------------------------------------------------------
        print("[1/7] fitter")
        f1 = tmp / "fit-a"
        p = run([str(HERE / "fit_dense_vq.py"), "--src", str(src), "--out", str(f1),
                 "--k", "16", "--dim", "2", "--layers", f"0-{LAYERS-1}",
                 "--family", "qwen3_8_dense", "--iters", "2", "--seed", "1234"], verbose=v)
        if not check("fit runs and writes an artifact", p.returncode == 0
                     and (f1 / "config.json").exists()):
            return report()
        check("fit reports SEEDED by default", "SEEDED" in p.stdout)
        cfg = json.load(open(f1 / "config.json"))
        nmod = len(cfg.get("vq_modules", {}))
        check("every target tensor was fit", nmod == LAYERS * len(PROJS),
              f"{nmod} modules")
        check("code dtype follows K (uint8 at K<=256)",
              mx.load(str(f1 / "model-00001-of-00001.safetensors"))[
                  list(cfg["vq_modules"])[0] + ".codes"].dtype == mx.uint8)

        # seed gate, BOTH directions (III.5)
        f2, f3 = tmp / "fit-b", tmp / "fit-unseeded"
        run([str(HERE / "fit_dense_vq.py"), "--src", str(src), "--out", str(f2),
             "--k", "16", "--dim", "2", "--layers", f"0-{LAYERS-1}",
             "--family", "qwen3_8_dense", "--iters", "2", "--seed", "1234"], verbose=v)
        run([str(HERE / "fit_dense_vq.py"), "--src", str(src), "--out", str(f3),
             "--k", "16", "--dim", "2", "--layers", f"0-{LAYERS-1}",
             "--family", "qwen3_8_dense", "--iters", "2", "--seed", "-1"], verbose=v)
        A, B, C = (decode_all(x) for x in (f1, f2, f3))
        m0 = list(A)[0]
        same_seed = float(mx.mean(mx.abs(A[m0] - B[m0])).item())
        diff_seed = float(mx.mean(mx.abs(A[m0] - C[m0])).item())
        check("same seed reproduces the fit", same_seed < 1e-6,
              f"mean|delta| {same_seed:.2e}")
        check("--seed -1 gives an independent draw (gate fails on known-bad)",
              diff_seed > same_seed * 10, f"mean|delta| {diff_seed:.2e}")

        # ---------------------------------------------------------------
        print("[2/7] outlier gate")
        p = run([str(HERE / "verify_artifact.py"), "--artifact", str(f1),
                 "--src", str(src), "--family", "qwen3_8_dense",
                 "--outlier", "3.0"], verbose=v)
        check("gate PASSES a healthy artifact", p.returncode == 0)

        bad = tmp / "fit-corrupt"
        shutil.copytree(f1, bad)
        sh = bad / "model-00001-of-00001.safetensors"
        # Read EAGERLY on the cpu stream before writing the same path back.
        # A lazy mx.load left unevaluated is materialized inside the save's
        # command buffer — i.e. read from a file already being overwritten,
        # which scrambles every tensor, not just the one being corrupted
        # (FINDINGS IV.1; hit while writing this very test).
        with mx.stream(mx.cpu):
            w = dict(mx.load(str(sh)))
            mx.eval(list(w.values()))
        w[m0 + ".codebook"] = mx.zeros_like(w[m0 + ".codebook"])  # collapse one
        mx.save_safetensors(str(sh), w)
        p = run([str(HERE / "verify_artifact.py"), "--artifact", str(bad),
                 "--src", str(src), "--family", "qwen3_8_dense",
                 "--outlier", "3.0"], expect_rc=1, verbose=v)
        check("gate FAILS a collapsed tensor (known-bad)", p.returncode != 0)

        # ---------------------------------------------------------------
        print("[3/7] packer")
        packed = tmp / "packed"
        p = run([str(HERE / "pack_dense.py"), "--src", str(f1), "--out", str(packed)],
                verbose=v)
        if check("pack runs", p.returncode == 0):
            u = sum(f.stat().st_size for f in f1.glob("*.safetensors"))
            q = sum(f.stat().st_size for f in packed.glob("*.safetensors"))
            check("packed artifact is smaller than unpacked", q < u,
                  f"{q} < {u} bytes")
            D_ = decode_all(packed)
            worst = max(float(mx.max(mx.abs(A[k] - D_[k])).item()) for k in A)
            check("packing is bit-exact (decode unchanged)", worst == 0.0,
                  f"max|delta| {worst}")

        # ---------------------------------------------------------------
        print("[4/7] provenance manifest")
        mdir = tmp / "manifests"
        env_run = lambda args, rc=0: subprocess.run(
            [PY, *args], capture_output=True, text=True,
            env={**__import__("os").environ, "VQLAB_MANIFEST_DIR": str(mdir)})
        p = env_run([str(HERE / "artifact_manifest.py"), "write", str(packed)])
        check("manifest write", p.returncode == 0 and any(mdir.glob("*.json")))
        p = env_run([str(HERE / "artifact_manifest.py"), "check", str(packed)])
        check("manifest check PASSES untouched bytes", p.returncode == 0)
        tgt = next(packed.glob("*.safetensors"))
        tgt.write_bytes(tgt.read_bytes() + b"\0")  # change size => identity breaks
        p = env_run([str(HERE / "artifact_manifest.py"), "check", str(packed)])
        check("manifest check FAILS altered bytes (known-bad)", p.returncode != 0)

        # ---------------------------------------------------------------
        print("[5/7] bundle gate")
        for nm, files, dense, want in (
                ("moe-good", ["vq_switch.py"], False, 0),
                ("moe-stale", [], False, 1),
                ("dense-good", ["vq_switch.py", "vq_dense.py"], True, 0),
                ("dense-missing-switch", ["vq_dense.py"], True, 1)):
            d = tmp / ("bundle-" + nm)
            d.mkdir()
            json.dump({"vq_linear" if dense else "vq_modules": {"x": {}}},
                      open(d / "config.json", "w"))
            (d / "model.py").write_text(
                "".join((HERE / f).read_text() for f in files) or "# stale\n")
            p = run([str(HERE / "check_bundle.py"), "--artifact", str(d)],
                    expect_rc=want, verbose=v)
            check(f"check-bundle {nm} -> {'PASS' if want == 0 else 'FAIL'}",
                  p.returncode == want)

        # ---------------------------------------------------------------
        print("[6/7] runtime kernels")
        bundle_txt = ((HERE / "vq_switch.py").read_text()
                      + (HERE / "vq_dense.py").read_text())
        ns = {"__name__": "bundled_model"}
        exec(compile(bundle_txt, "model.py", "exec"), ns)
        K_, D_d = 16, 2
        codes = mx.random.randint(0, K_, (OUT_D, IN_D // D_d)).astype(mx.uint8)
        cb = (mx.random.normal((K_, D_d)) * 0.1).astype(mx.float16)
        sc = (mx.random.uniform(shape=(OUT_D, IN_D // G)) * .5 + .5).astype(mx.float16)
        x = (mx.random.normal((1, IN_D)) * 0.5).astype(mx.float16)
        yb = ns["VQLinear"](codes, cb, sc, group_size=G, pack_bits=0)(x)
        mx.eval(yb)
        sys.path.insert(0, str(HERE))
        import vq_dense
        yi = vq_dense.VQLinear(codes, cb, sc, group_size=G, pack_bits=0)(x)
        mx.eval(yi)
        check("bundled and installed runtimes agree bit-for-bit",
              bool(mx.array_equal(yb, yi)))

        # Kernel RESOLUTION: name which copy each path actually uses
        # (III.13 — never assume). The bundle must resolve from its own
        # globals; the standalone module must resolve to the package sibling,
        # NOT to mlx_lm.models.vq_switch, which exists only on VQ-patched
        # lab machines. The original version of this check asserted the
        # standalone path FAILS without mlx_lm — true of the old fallback
        # chain, fixed the first time the selftest ran in a fresh venv.
        bundled_fn = ns.get("_dense_fused")
        check("bundle resolves _dense_fused from its own globals",
              bundled_fn is not None)
        resolved = vq_dense._resolve_kernel("_dense_fused")
        rmod = getattr(resolved, "__module__", "?")
        check("standalone resolves the sibling, not a patched mlx_lm",
              "mlx_lm" not in rmod, f"resolved from {rmod}")

        # ---------------------------------------------------------------
        print("[7/7] pricer")
        p = run([str(HERE / "price.py"), "--family", "qwen397b",
                 "--budget-gib", "108"], verbose=v)
        check("price emits a recipe for a byte budget",
              p.returncode == 0 and "harvest recipe" in p.stdout)

        skip("end-to-end generation smoke",
             "needs a real checkpoint + mlx-lm architecture; run "
             "`vqlab smoke` on a real artifact")
        skip("scoring (referee / KL)",
             "needs a real model and teacher cache")
        return report()
    finally:
        if a.keep:
            print(f"\nworkspace kept: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


def report() -> int:
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
