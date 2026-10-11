"""Selftest: the records (provenance manifest, build records, lineage, the fit
store's recipes, records written through the CLI)."""
from __future__ import annotations

import json
import os
import subprocess

from vqlab import _layout
from vqlab._layout import find as _find
from vqlab.selftest import PY, check, run
from vqlab.selftest.synthetic_model import LAYERS, PROJS, geo_fixture


def check_manifest(ctx):
    print("[4/7] provenance manifest")
    packed = ctx.packed
    mdir = ctx.tmp / "manifests"
    env_run = lambda args, rc=0: subprocess.run(
        [PY, *args], capture_output=True, text=True,
        env={**os.environ, "VQLAB_MANIFEST_DIR": str(mdir)})
    p = env_run([str(_find("artifact_manifest.py")), "write", str(packed)])
    check("manifest write", p.returncode == 0 and any(mdir.glob("*.json")))
    p = env_run([str(_find("artifact_manifest.py")), "check", str(packed)])
    check("manifest check PASSES untouched bytes", p.returncode == 0)
    tgt = next(packed.glob("*.safetensors"))
    tgt.write_bytes(tgt.read_bytes() + b"\0")  # change size => identity breaks
    p = env_run([str(_find("artifact_manifest.py")), "check", str(packed)])
    check("manifest check FAILS altered bytes (known-bad)", p.returncode != 0)


def check_build_records(ctx):
    print("[4b/7] build records (vqlab provenance)")
    _check_fit_dense_record(ctx)
    ctx.geo = g = geo_fixture(ctx.tmp)
    p = _geo_build(ctx, g["out"])
    if check("geo-build runs on the fixture", p.returncode == 0):
        _check_geo_build_record(ctx)
        _check_geo_build_pool(ctx)
        _check_lineage_and_verify(ctx)


def _geo_build(ctx, out, *extra):
    g = ctx.geo
    return run([str(_find("geo_build.py")), "--artifact", str(g["base"]),
                "--teacher", str(g["teacher"]), "--family", "qwen3_5",
                "--geomap", str(g["map"]), "--out", str(out), *extra,
                "--memory-limit-gb", "4"], verbose=ctx.verbose)


def _check_fit_dense_record(ctx):
    rec = json.load(open(ctx.f1 / "vqlab_provenance.json"))
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
          json.load(open(ctx.f3 / "vqlab_provenance.json"))["method"]["seed"]
          == "unseeded")
    p = run([str(_find("provenance.py")), str(ctx.f1), "--verify"], verbose=ctx.verbose)
    check("provenance --verify PASSES untouched build", p.returncode == 0)


def _check_geo_build_record(ctx):
    g = ctx.geo
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
    out2 = ctx.tmp / "geo-out2"
    _geo_build(ctx, out2, "--parts", str(g["out"]) + "_parts")
    r2 = json.load(open(out2 / "vqlab_provenance.json"))
    check("origin survives a resume (ledger, not guesswork)",
          r2["modules"][g["module"]]["origin"] == "fit")
    from vqlab.core.source import fit_store as fs_
    stored = fs_.read_index(ctx.tmp / "fitstore")
    geo_fits = [r for r in stored.values() if r["module"] == g["module"]]
    check("geo-build files its fit in the store, recipe attached",
          len(geo_fits) == 1 and geo_fits[0]["recipe"]["origin"] == "fit"
          and geo_fits[0]["recipe"]["fitter"]["alternation"] is True,
          f"{len(geo_fits)} stored")
    cu = fs_.code_usage(geo_fits[0])
    gf = geo_fits[0]
    check("fits census counts every code exactly (unpacked from the bit-packed codes)",
          cu["exact"] and cu["lookups"] == gf["E"] * gf["OUT"] * (gf["IN"] // gf["d"])
          and 0 <= cu["dead"] <= gf["K"] and 0 <= cu["zero_scale_frac"] <= 1,
          f"lookups={cu['lookups']}")


def _check_geo_build_pool(ctx):
    g = ctx.geo
    out3 = ctx.tmp / "geo-out3"
    _geo_build(ctx, out3, "--pool")
    r3 = json.load(open(out3 / "vqlab_provenance.json"))["modules"][g["module"]]
    same = all((out3 / f.name).read_bytes() == f.read_bytes()
               for f in g["out"].glob("*.safetensors") if not f.is_symlink())
    check("--pool reuses the stored fit (no refit) and rebuilds byte-identical",
          r3["origin"] == "reuse" and r3["source_origin"]["origin"] == "store"
          and same, f"origin={r3['origin']}")
    out4 = ctx.tmp / "geo-out4"
    _geo_build(ctx, out4, "--pool", "--tail-weight-pow", "1.0")
    r4 = json.load(open(out4 / "vqlab_provenance.json"))["modules"][g["module"]]
    check("--pool never reuses a fit of ANOTHER recipe (plain fit, tail-weighted build)",
          r4["origin"] == "fit", f"origin={r4['origin']}")


def _check_lineage_and_verify(ctx):
    g = ctx.geo
    prov = _find("provenance.py")
    from vqlab.records.provenance import measured as _meas
    mo = _meas(g["out"])
    check("scores stamp what they measured: build record id + byte fingerprint",
          mo["build_record"] == json.load(open(g["out"] / "vqlab_provenance.json"))["id"]
          and len(mo["fingerprint"]) == 16)
    p = run([str(prov), str(g["out"]), "--lineage"], verbose=ctx.verbose)
    check("provenance --lineage walks to the base",
          p.returncode == 0 and "fit-dense" not in p.stdout
          and str(g["base"]) in p.stdout)
    shard = next(f for f in g["out"].glob("*.safetensors")
                 if not f.is_symlink())
    shard.write_bytes(shard.read_bytes() + b"\0")
    p = run([str(prov), str(g["out"]), "--verify"], expect_rc=2,
            verbose=ctx.verbose)
    check("provenance --verify FAILS a rewritten shard (known-bad)",
          p.returncode == 2)


def check_cli_build_records(ctx):
    print("[5c/7] CLI build records: fresh output + in-place amendment")
    g = ctx.geo
    cenv = {**os.environ, "PYTHONPATH": str(_layout.SRC)}
    cli = lambda *a_: subprocess.run([PY, "-m", "vqlab.cli", *a_],
                                     capture_output=True, text=True, env=cenv)
    p2 = ctx.tmp / "packed-cli"
    cli("pack", "--src", str(ctx.f1), "--out", str(p2))      # a raw fit's vq_modules (see [3/7])
    r2 = json.load(open(p2 / "vqlab_provenance.json")) if (p2 / "vqlab_provenance.json").exists() else {}
    check("pack via the CLI gets a build record linked to its fit",
          r2.get("tool", {}).get("name") == "pack"
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
