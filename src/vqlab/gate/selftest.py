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
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # src/
from vqlab._layout import find as _find  # noqa: E402
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


def geo_fixture(tmp: pathlib.Path):
    """A qwen3_5-shaped MoE teacher + a VQ base artifact for geo-build: one
    module, E=2 experts, IN=128 so d4 gives nsub=32 (geo-build refuses
    ragged packing). The base carries its OWN build record so the child's
    lineage link can be checked."""
    import provenance
    E, I, H = 2, 8, 128
    teacher, base = tmp / "geo-teacher", tmp / "geo-base"
    teacher.mkdir(); base.mkdir()
    mx.random.seed(11)
    tk = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    mx.save_safetensors(str(teacher / "t.safetensors"),
                        {tk: (mx.random.normal((E, 2 * I, H)) * .05).astype(mx.bfloat16)})
    json.dump({"weight_map": {tk: "t.safetensors"}},
              open(teacher / "model.safetensors.index.json", "w"))
    mod = "model.language_model.layers.0.mlp.switch_mlp.gate_proj"
    w = {mod + ".codes": mx.zeros((E, I, H // 2), mx.uint8),
         mod + ".codebook": mx.zeros((256, 2), mx.float16),
         mod + ".vq_scales": mx.ones((E, I, H // G), mx.float16)}
    mx.save_safetensors(str(base / "model-00001-of-00001.safetensors"), w)
    json.dump({"weight_map": {k: "model-00001-of-00001.safetensors" for k in w}},
              open(base / "model.safetensors.index.json", "w"))
    json.dump({"model_type": "qwen3_5_moe", "vq_modules": {mod: {
        "experts": E, "out": I, "in": H, "group": G, "dim": 2, "k": 256}}},
        open(base / "config.json", "w"))
    brec = provenance.write_build_record(base, tool="selftest-fixture")
    gm = tmp / "geomap.json"
    json.dump({mod: {"dim": 4, "k": 16}}, open(gm, "w"))
    return {"teacher": teacher, "base": base, "map": gm, "module": mod,
            "out": tmp / "geo-out", "base_id": brec["id"]}


def decode_all(art: pathlib.Path):
    """Decode every VQ tensor in an artifact to dense weights."""
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
    # Every fitter files into the fit store; point it (and the family data
    # dir) at the workspace so a selftest never writes to a real store.
    __import__("os").environ["VQLAB_FIT_STORE"] = str(tmp / "fitstore")
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
        p = run([str(_find("fit_dense_vq.py")), "--src", str(src), "--out", str(f1),
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
        run([str(_find("fit_dense_vq.py")), "--src", str(src), "--out", str(f2),
             "--k", "16", "--dim", "2", "--layers", f"0-{LAYERS-1}",
             "--family", "qwen3_8_dense", "--iters", "2", "--seed", "1234"], verbose=v)
        run([str(_find("fit_dense_vq.py")), "--src", str(src), "--out", str(f3),
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
        p = run([str(_find("verify_artifact.py")), "--artifact", str(f1),
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
        p = run([str(_find("verify_artifact.py")), "--artifact", str(bad),
                 "--src", str(src), "--family", "qwen3_8_dense",
                 "--outlier", "3.0"], expect_rc=1, verbose=v)
        check("gate FAILS a collapsed tensor (known-bad)", p.returncode != 0)

        # ---------------------------------------------------------------
        print("[3/7] packer")
        packed = tmp / "packed"
        p = run([str(_find("pack_dense.py")), "--src", str(f1), "--out", str(packed)],
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
        p = env_run([str(_find("artifact_manifest.py")), "write", str(packed)])
        check("manifest write", p.returncode == 0 and any(mdir.glob("*.json")))
        p = env_run([str(_find("artifact_manifest.py")), "check", str(packed)])
        check("manifest check PASSES untouched bytes", p.returncode == 0)
        tgt = next(packed.glob("*.safetensors"))
        tgt.write_bytes(tgt.read_bytes() + b"\0")  # change size => identity breaks
        p = env_run([str(_find("artifact_manifest.py")), "check", str(packed)])
        check("manifest check FAILS altered bytes (known-bad)", p.returncode != 0)

        # ---------------------------------------------------------------
        print("[4b/7] build records (vqlab provenance)")
        prov = _find("provenance.py")
        rec = json.load(open(f1 / "vqlab_provenance.json"))
        check("fit-dense writes a build record",
              rec["tool"]["name"] == "fit-dense" and rec["method"]["seed"] == 1234
              and rec["inputs"][0]["role"] == "source")
        check("record names every module with its origin",
              len(rec["modules"]) == LAYERS * len(PROJS)
              and all(m["origin"] == "fit" for m in rec["modules"].values()))
        check("record lists args left at default", "init_cap" in
              rec["tool"]["args"]["at_default"] and "k" not in
              rec["tool"]["args"]["at_default"])
        check("unseeded fit is recorded as unseeded",
              json.load(open(f3 / "vqlab_provenance.json"))["method"]["seed"]
              == "unseeded")
        p = run([str(prov), str(f1), "--verify"], verbose=v)
        check("provenance --verify PASSES untouched build", p.returncode == 0)
        g = geo_fixture(tmp)
        p = run([str(_find("geo_build.py")), "--artifact", str(g["base"]),
                 "--teacher", str(g["teacher"]), "--family", "qwen3_5",
                 "--geomap", str(g["map"]), "--out", str(g["out"]),
                 "--memory-limit-gb", "4"], verbose=v)
        if check("geo-build runs on the fixture", p.returncode == 0):
            grec = json.load(open(g["out"] / "vqlab_provenance.json"))
            mod = grec["modules"][g["module"]]
            check("geo-build record: origin fit, alternation ON by default",
                  mod["origin"] == "fit" and grec["method"]["alternation"] is True
                  and "plain_lloyd" in grec["tool"]["args"]["at_default"])
            check("geo-build record links its base's record (lineage)",
                  grec["inputs"][0]["role"] == "base"
                  and grec["inputs"][0]["provenance_id"] == g["base_id"])
            # resume: rerun into a fresh out with the same parts dir -- the
            # module is not refit, and the ledger must still say "fit"
            out2 = tmp / "geo-out2"
            run([str(_find("geo_build.py")), "--artifact", str(g["base"]),
                 "--teacher", str(g["teacher"]), "--family", "qwen3_5",
                 "--geomap", str(g["map"]), "--out", str(out2),
                 "--parts", str(g["out"]) + "_parts",
                 "--memory-limit-gb", "4"], verbose=v)
            r2 = json.load(open(out2 / "vqlab_provenance.json"))
            check("origin survives a resume (ledger, not guesswork)",
                  r2["modules"][g["module"]]["origin"] == "fit")
            import importlib as _il
            fs_ = _il.import_module("fitstore")
            stored = fs_.read_index(tmp / "fitstore")
            geo_fits = [r for r in stored.values() if r["module"] == g["module"]]
            check("geo-build files its fit in the store, recipe attached",
                  len(geo_fits) == 1 and geo_fits[0]["recipe"]["origin"] == "fit"
                  and geo_fits[0]["recipe"]["fitter"]["alternation"] is True,
                  f"{len(geo_fits)} stored")
            out3 = tmp / "geo-out3"
            run([str(_find("geo_build.py")), "--artifact", str(g["base"]),
                 "--teacher", str(g["teacher"]), "--family", "qwen3_5",
                 "--geomap", str(g["map"]), "--out", str(out3), "--pool",
                 "--memory-limit-gb", "4"], verbose=v)
            r3 = json.load(open(out3 / "vqlab_provenance.json"))["modules"][g["module"]]
            same = all((out3 / f.name).read_bytes() == f.read_bytes()
                       for f in g["out"].glob("*.safetensors") if not f.is_symlink())
            check("--pool reuses the stored fit (no refit) and rebuilds byte-identical",
                  r3["origin"] == "reuse" and r3["source_origin"]["origin"] == "store"
                  and same, f"origin={r3['origin']}")
            p = run([str(prov), str(g["out"]), "--lineage"], verbose=v)
            check("provenance --lineage walks to the base",
                  p.returncode == 0 and "fit-dense" not in p.stdout
                  and str(g["base"]) in p.stdout)
            shard = next(f for f in g["out"].glob("*.safetensors")
                         if not f.is_symlink())
            shard.write_bytes(shard.read_bytes() + b"\0")
            p = run([str(prov), str(g["out"]), "--verify"], expect_rc=2,
                    verbose=v)
            check("provenance --verify FAILS a rewritten shard (known-bad)",
                  p.returncode == 2)

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
                "".join((_find(f)).read_text() for f in files) or "# stale\n")
            p = run([str(_find("check_bundle.py")), "--artifact", str(d)],
                    expect_rc=want, verbose=v)
            check(f"check-bundle {nm} -> {'PASS' if want == 0 else 'FAIL'}",
                  p.returncode == want)

        # ---------------------------------------------------------------
        print("[6/7] runtime kernels")
        bundle_txt = ((_find("vq_switch.py")).read_text()
                      + (_find("vq_dense.py")).read_text())
        ns = {"__name__": "bundled_model"}
        exec(compile(bundle_txt, "model.py", "exec"), ns)
        K_, D_d = 16, 2
        codes = mx.random.randint(0, K_, (OUT_D, IN_D // D_d)).astype(mx.uint8)
        cb = (mx.random.normal((K_, D_d)) * 0.1).astype(mx.float16)
        sc = (mx.random.uniform(shape=(OUT_D, IN_D // G)) * .5 + .5).astype(mx.float16)
        x = (mx.random.normal((1, IN_D)) * 0.5).astype(mx.float16)
        yb = ns["VQLinear"](codes, cb, sc, group_size=G, pack_bits=0)(x)
        mx.eval(yb)
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
        print("[5b/7] fit-moe: files its fits, writes a build record")
        mt, mb, mo = tmp / "moe-teacher", tmp / "moe-base", tmp / "moe-out"
        mt.mkdir(); mb.mkdir()
        E_, I_, H_ = 2, 8, 128
        tk = "model.language_model.layers.0.mlp.experts.gate_up_proj"
        mx.save_safetensors(str(mt / "t.safetensors"),
                            {tk: (mx.random.normal((E_, 2 * I_, H_)) * .05).astype(mx.bfloat16)})
        json.dump({"weight_map": {tk: "t.safetensors"}},
                  open(mt / "model.safetensors.index.json", "w"))
        bm = "model.language_model.layers.0.mlp.switch_mlp.gate_proj"
        bw = {bm + ".weight": mx.zeros((E_, I_, H_ * 2 // 32), mx.uint32),
              bm + ".scales": mx.ones((E_, I_, H_ // G), mx.float16),
              bm + ".biases": mx.zeros((E_, I_, H_ // G), mx.float16)}
        mx.save_safetensors(str(mb / "model-00001-of-00001.safetensors"), bw)
        json.dump({"weight_map": {k: "model-00001-of-00001.safetensors" for k in bw}},
                  open(mb / "model.safetensors.index.json", "w"))
        json.dump({"model_type": "qwen3_5_moe",
                   "quantization": {"group_size": G, "bits": 4,
                                    bm: {"group_size": G, "bits": 2}}},
                  open(mb / "config.json", "w"))
        p = run([str(_find("vq_397b_codes.py")), "--base", str(mb), "--src", str(mt),
                 "--out", str(mo), "--vq-layers", "0", "--k", "16", "--dim", "4",
                 "--iters", "2", "--sample", "1000", "--family", "qwen3_5",
                 "--relerr-abort", "1.0"], verbose=v)
        if check("fit-moe runs on the fixture", p.returncode == 0,
                 (p.stderr or p.stdout)[-200:] if p.returncode else ""):
            mrec = json.load(open(mo / "vqlab_provenance.json"))
            check("fit-moe build record: method + module origin",
                  mrec["tool"]["name"] == "fit-moe" and mrec["method"]["seed"] == 1234
                  and mrec["modules"][bm]["origin"] == "fit")
            fs_ = __import__("importlib").import_module("fitstore")
            got = [r for r in fs_.read_index(tmp / "fitstore").values()
                   if r["module"] == bm and r["recipe"]["tool"] == "fit-moe"]
            check("fit-moe files its fit in the store with its recipe",
                  len(got) == 1 and got[0]["d"] == 4 and got[0]["K"] == 16
                  and got[0]["teacher"] == "moe-teacher", f"{len(got)} stored")

        print("[5c/7] CLI build records: fresh output + in-place amendment")
        cenv = {**__import__("os").environ, "PYTHONPATH": str(HERE.parents[1])}
        cli = lambda *a_: subprocess.run([PY, "-m", "vqlab.cli", *a_],
                                         capture_output=True, text=True, env=cenv)
        p2 = tmp / "packed-cli"
        cli("pack-dense", "--src", str(f1), "--out", str(p2))
        r2 = json.load(open(p2 / "vqlab_provenance.json")) if (p2 / "vqlab_provenance.json").exists() else {}
        check("pack-dense via the CLI gets a build record linked to its fit",
              r2.get("tool", {}).get("name") == "pack-dense"
              and any(i.get("provenance_id") for i in r2.get("inputs", [])))
        if (g["out"] / "vqlab_provenance.json").exists():
            before = json.load(open(g["out"] / "vqlab_provenance.json"))["id"]
            # geo-out was corrupted by the known-bad --verify check above; the
            # amendment must still record what it was and what it is now
            cli("bundle", "--artifact", str(g["out"]), "--group", "64")
            ra = json.load(open(g["out"] / "vqlab_provenance.json"))
            hist = (g["out"] / "vqlab_provenance.history.jsonl").read_text().splitlines()
            check("in-place bundle AMENDS: new record links the old, history kept",
                  ra["tool"]["name"] == "bundle" and ra.get("previous") == before
                  and json.loads(hist[-1])["id"] == before)

        print("[6a/7] family-profile: unknown family -> data entry, no code change")
        U = tmp / "novel-teacher"
        U.mkdir()
        wn = {f"transformer.blocks.{li}.moe.experts.{p_}.weight":
              mx.zeros((4, 64, 128), mx.bfloat16)
              for li in range(2) for p_ in ("w1", "w2", "w3")}
        mx.save_safetensors(str(U / "model.safetensors"), wn)
        json.dump({"model_type": "novelmoe", "num_hidden_layers": 2,
                   "num_experts": 4, "hidden_size": 128}, open(U / "config.json", "w"))
        fenv = {**__import__("os").environ, "VQLAB_FAMILIES_DIR": str(tmp / "fams")}
        fp_ = [PY, str(_find("family_profile.py")), "--teacher", str(U),
               "--out", str(tmp / "novel-prof")]
        p = subprocess.run(fp_, capture_output=True, text=True, env=fenv)
        check("unknown family is reported UNKNOWN, not mis-matched (known-bad)",
              "UNKNOWN" in p.stdout and json.load(open(
                  tmp / "novel-prof" / "profile.json")).get("unknown_family") is True,
              p.stdout.splitlines()[0] if p.stdout else p.stderr[-200:])
        p = subprocess.run(fp_ + ["--write-entry", "novelmoe"], capture_output=True,
                           text=True, env=fenv)
        p = subprocess.run(fp_, capture_output=True, text=True, env=fenv)
        prof = json.load(open(tmp / "novel-prof" / "profile.json"))
        check("drafted data entry makes the family profile: 6 modules found",
              prof["family"] == "novelmoe" and prof["modules"]["count"] == 6,
              f"family={prof['family']} modules={prof['modules']['count']}")

        print("[6a2/7] onboard: sequences the steps, resumes, never fakes a GPU step")
        ob = [PY, str(_find("onboard.py")), "--teacher", str(U)]
        oenv = {**fenv, "VQLAB_SCRATCH": str(tmp / "scratch")}
        p = subprocess.run(ob, capture_output=True, text=True, env=oenv)
        stf = tmp / "fams" / "novelmoe" / "teachers" / "novel-teacher" / "onboard.json"
        ost = json.load(open(stf))["steps"] if stf.exists() else {}
        check("onboard runs the CPU steps and stops at the first GPU step",
              ost.get("profile", {}).get("status") == "done"
              and ost.get("loader", {}).get("status") == "done"
              and ost.get("cache_a", {}).get("status") == "pending"
              and "kl cache" in ost.get("cache_a", {}).get("result", {}).get("command", ""),
              (p.stdout or p.stderr)[-200:])
        p = subprocess.run(ob, capture_output=True, text=True, env=oenv)
        ost2 = json.load(open(stf))["steps"]
        check("rerunning onboard resumes (profile/loader stay done, nothing launched)",
              ost2["loader"]["status"] == "done" and ost2["cache_a"]["status"] == "pending"
              and "run_id" not in ost2["cache_a"]["result"])

        print("[6a3/7] pin + step verdict: scorers refuse what was not smoked")
        from vqlab.gate import pin as P
        from vqlab.records.step_verdict import step_verdict
        psrc = tmp / "pin_src"
        make_source(psrc)
        (psrc / "model.py").write_text("# fixture runtime\n")
        pout = tmp / "pin_out"
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            prc = P.pin(psrc, pout, headroom=1e-12)   # forces the RAM refusal
        prec = json.load(open(pout / P.MARKER))
        check("pin: weights symlinked to the resolved source, metadata copied",
              (pout / "model-00001-of-00001.safetensors").is_symlink()
              and not (pout / "config.json").is_symlink())
        check("pin: too big for the box -> deferred-ram, allowed with a WARNING",
              prc == 0 and prec["state"] == "deferred-ram"
              and P.check_pin(pout)[0] and "WARNING" in P.check_pin(pout)[1])
        try:
            P.pin(psrc, pout)
            refused_overwrite = False
        except SystemExit:
            refused_overwrite = True
        check("pin: never overwrites an existing directory", refused_overwrite)
        (pout / "model.py").write_text("# rebundled after the smoke\n")
        check("pin: model.py changed after pinning -> refused", not P.check_pin(pout)[0])
        prec["state"], prec["model_py_sha256"] = "failed", P._sha(pout / "model.py")
        (pout / P.MARKER).write_text(json.dumps(prec))
        check("pin: failed smoke -> refused", not P.check_pin(pout)[0])
        check("pin: an unpinned directory is allowed unchanged", P.check_pin(psrc) == (True, ""))
        so, se = tmp / "step.out", tmp / "step.err"
        so.write_text("")
        se.write_text("Traceback (most recent call last):\n  File x\nFileNotFoundError: corpus\n")
        v = step_verdict(0, so, se, {"stdout_regex": "^{", "min_lines": 1})
        check("step_verdict: rc 0 + traceback + no output is a FAIL (the night-4 speed bug)",
              not v["ok"] and "FileNotFoundError" in v["stderr_tail"][-1])
        so.write_text('{"arm": "a"}\n')
        se.write_text("")
        check("step_verdict: rc 0 + expected output passes",
              step_verdict(0, so, se, {"stdout_regex": "^{"})["ok"])
        check("step_verdict: nonzero exit fails", not step_verdict(3, so, se)["ok"])

        print("[6a4/7] queue: pinned tree, loud failure, preflight")
        import os as _os
        from vqlab.agents import run_queue as Q
        from vqlab import _layout as L
        _saved = {k: _os.environ.get(k) for k in ("VQLAB_QUEUE_DIR", "VQLAB_GPU_LEASE",
                                                   "VQLAB_PREFLIGHT_DIR")}
        _os.environ["VQLAB_QUEUE_DIR"] = str(tmp / "queues")
        _os.environ["VQLAB_PREFLIGHT_DIR"] = str(tmp / "pf")
        _os.environ["VQLAB_GPU_LEASE"] = str(tmp / "gpu.lease")
        qf = tmp / "q.json"
        qf.write_text(json.dumps({"name": "st", "steps": [
            {"name": "ok", "cmd": "runs", "args": ["-n", "1"], "preflight": {"append": []}},
            {"name": "gone", "cmd": "fits", "args": ["list", "--root", str(tmp / "no" / "such")]},
            {"name": "after", "cmd": "runs", "args": ["-n", "1"]}]}))
        qdirs = []
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                qd = Q.create(qf, allow_dirty=True)
                qdirs.append(qd)
                qrc = Q.run(qd)
            qs = json.load(open(qd / "state.json"))
            check("queue: steps run from a worktree pinned at the queue's commit",
                  Q._git("rev-parse", "HEAD", cwd=qd / "tree") == qs["commit"]
                  and str(qd / "tree") in (qd / "steps" / "00-ok" / "cmd").read_text() + qs["tree"])
            check("queue: a missing input path FAILS the step and stops the queue",
                  qrc == 4 and [r["status"] for r in qs["steps"]] == ["pass", "fail", "pending"]
                  and "does not exist" in qs["steps"][1]["reasons"][0])
            with contextlib.redirect_stdout(io.StringIO()):
                pd = Q.create(qf, allow_dirty=True, preflight=True)
                qdirs.append(pd)
                Q.run(pd)
            ps_ = json.load(open(pd / "state.json"))
            check("queue --preflight: reports every step; no preflight block is a FAIL",
                  [r["status"] for r in ps_["steps"]] == ["pass", "fail", "fail"]
                  and "no preflight defined" in ps_["steps"][2]["reasons"][0])
            om = {}
            r1 = Q._redirect_outs(["--teacher", "/T", "--out", "/V/fit"], tmp / "pf" / "a", om)
            r2 = Q._redirect_outs(["--rung", "r1=/V/fit", "/V/fit/x", "--out=/V/pin"],
                                  tmp / "pf" / "b", om)
            check("queue --preflight: outputs redirected off the real paths; chained inputs follow",
                  r1[3] == str(tmp / "pf" / "a" / "fit")
                  and r2[:3] == ["--rung", f"r1={tmp}/pf/a/fit", f"{tmp}/pf/a/fit/x"]
                  and r2[3] == f"--out={tmp}/pf/b/pin")
            check("queue: name=path args are checked as paths",
                  Q._path_args(["--cache", "prose=/V/c", "--out", "/o"]) == (["/V/c"], ["/o"]))
            check("queue: publish can never be queued",
                  any("publish" in e for e in Q.validate({"steps": [{"name": "p", "cmd": "publish"}]})))
        finally:
            for qd in qdirs:
                subprocess.run(["git", "worktree", "remove", "--force", str(qd / "tree")],
                               cwd=str(L.SRC.parent), capture_output=True)
            for k, v in _saved.items():
                if v is None:
                    _os.environ.pop(k, None)
                else:
                    _os.environ[k] = v
        check("corpus locator: prose / code / lit by name, old spellings too",
              all(L.corpus(n).is_file() for n in ("prose", "code", "lit", "wikitext"))
              and L.corpus(str(L.SRC / "vqlab" / "referee" / "referee_corpus.txt")).is_file())

        print("[6b/7] layout")
        # A stage module whose bare name is also a stdlib or installed
        # package gets shadowed (or shadows it) on sys.path. bench/coverage.py
        # was renamed kernel_coverage.py for exactly this (pytest-cov).
        from vqlab import _layout as L
        import importlib.machinery as _im
        # Search only OUTSIDE this repo (site-packages, stdlib); find_spec
        # would also report modules we have already imported ourselves.
        repo = str(L.SRC.parent)
        outside = [p for p in sys.path if p and not p.startswith(repo)]
        clash = sorted(f.stem for d in L.stage_dirs() for f in d.glob("*.py")
                       if f.stem != "__init__" and (
                           f.stem in sys.stdlib_module_names
                           or _im.PathFinder.find_spec(f.stem, outside) is not None))
        import importlib as _il
        import vq_switch as _bare
        import vqlab.runtime.vq_switch as _canon
        import vqlab.vq_switch as _old
        check("bare, dotted and pre-split names give ONE module object",
              _bare is _canon is _old and _canon.__name__ == "vqlab.runtime.vq_switch")
        check("reload by the bare name re-executes the module",
              _il.reload(_bare) is _canon and _canon.__name__ == "vqlab.runtime.vq_switch")
        check("no stage module name collides with stdlib / site-packages",
              not clash, ", ".join(clash))
        p = subprocess.run([PY, "-m", "vqlab.price", "--help"], capture_output=True,
                           text=True, env={**__import__("os").environ,
                                           "PYTHONPATH": str(L.SRC)})
        check("pre-split `python -m vqlab.<name>` still runs", p.returncode == 0,
              p.stderr.strip().splitlines()[-1] if p.returncode else "")

        # ---------------------------------------------------------------
        print("[7/7] pricer")
        p = run([str(_find("price.py")), "--family", "qwen397b",
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
