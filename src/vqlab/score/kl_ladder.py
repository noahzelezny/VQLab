"""vqlab kl-ladder — rank a set of rungs by KL against per-corpus teacher caches.

THE INSTRUMENT THE 2026-09-16 FLASH-3.2 SWEEP DID NOT HAVE. That sweep
scored ppl on three corpora and KL on one, and the three ppl columns
disagreed with each other while the single KL column disagreed with all of
them: the arm with the best prose ppl was the arm KL said was worth nothing
(F116). Ppl is the release GATE; KL is the ranking instrument (quantlab III),
and a ranking instrument that only sees one corpus is not much of one.

    vqlab kl-ladder --cache prose=<dir> --cache code=<dir> --cache lit=<dir> \
        --rung shipped=<dir> --rung cand=<dir> [--out table.json]

Every rung is scored through `vqlab.stream_score` -- the SAME layer-streamed
path that carries the rule-5 validation -- once per cache, at the tokens and
chunk the cache itself records. One harness, no exceptions: a cache built at
chunk 512 scores students at chunk 512, because this family carries recurrent
state and the metric is not chunk-invariant (F111).

Output carries mean KL, its standard error over positions, and a pairwise
verdict against the first rung: SAME when the 95% intervals overlap. The
interval is a LOWER bound on uncertainty (positions are correlated), so an
overlap is a strong claim of no difference and a separation is a weak one.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # src/
from vqlab import _layout  # noqa: E402


def _kv(pairs, what):
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"FAIL: --{what} wants name=path, got {p!r}")
        k, v = p.split("=", 1)
        out[k] = v
    return out


def score_one(python, model, cache_dir, corpus, tokens, chunk, stream_ple,
              lazy_over_gb, per_pos=None, allow_build_mismatch=False):
    cmd = [python, "-m", "vqlab.stream_score", "--model", model,
           "--corpus", corpus, "--tokens", str(tokens), "--chunk", str(chunk),
           "--kl-cache", cache_dir, "--lazy-over-gb", str(lazy_over_gb)]
    if per_pos:
        cmd += ["--kl-per-position", per_pos]
    if stream_ple:
        cmd.append("--stream-ple")
    if allow_build_mismatch:
        cmd.append("--allow-build-mismatch")
    r = subprocess.run(cmd, capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    if not line:
        tail = (r.stderr or r.stdout or "").strip().splitlines()
        # LOUD, never a silent None: a rung that failed to score is not a
        # data point, and a table full of nulls looks exactly like a table.
        raise SystemExit(f"FAIL scoring {model} on {cache_dir} "
                         f"(rc={r.returncode}): "
                         + (tail[-1] if tail else "no output"))
    return json.loads(line[-1])


def interpreter_build(python, model_dir):
    """core.numerics.build as the SCORING interpreter resolves it (it may not
    be this one). Reads files and metadata only; None if it cannot run."""
    r = subprocess.run([python, "-m", "vqlab.core.numerics", "--model", model_dir],
                       capture_output=True, text=True)
    line = [x for x in r.stdout.splitlines() if x.startswith("{")]
    return json.loads(line[-1]) if r.returncode == 0 and line else None


# Preflight compares what does not depend on how a family resolves its
# runtime; the arch file is compared per cell by stream_score itself.
PREFLIGHT_FIELDS = ("mlx", "mlx_lm", "scorer_variant")


def _preflight_builds(python, rungs, meta, caches, allow):
    run = interpreter_build(python, next(iter(rungs.values())))
    if run is None:
        print("[kl-ladder] WARNING: could not read the scoring interpreter's numerics "
              "build; each cell is still checked by stream_score", flush=True)
        return None
    bad = []
    for name, d in caches.items():
        m = json.load(open(os.path.join(d, "meta.json")))
        cb = m.get("numerics")
        if cb is None:
            print(f"[kl-ladder] WARNING: cache {name} carries NO numerics build stamp "
                  f"(written before 2026-10-03); mlx/mlx-lm/arch cannot be checked", flush=True)
            cb = {"scorer_variant": m.get("scorer_variant")}
            fields = ("scorer_variant",)
        else:
            fields = PREFLIGHT_FIELDS
        bad += [f"{name}: {k} cache {cb.get(k)!r} vs run {run.get(k)!r}"
                for k in fields if cb.get(k) != run.get(k)]
    if bad and not allow:
        raise SystemExit("FAIL: cache(s) built with different numerics than the scoring "
                         "interpreter -- " + "; ".join(bad) + ". Rebuild them here, or pass "
                         "--allow-build-mismatch (recorded).")
    for b in bad:
        print(f"[kl-ladder] WARNING: build mismatch ALLOWED: {b}", flush=True)
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="append", required=True,
                    help="name=dir, repeatable (one per corpus)")
    ap.add_argument("--rung", action="append", required=True,
                    help="name=dir, repeatable; the FIRST is the reference")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--stream-ple", action="store_true")
    ap.add_argument("--lazy-over-gb", type=float, default=8.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--per-pos-dir", default=None,
                    help="where per-position KL arrays are kept (they are what "
                         "makes the PAIRED comparison possible)")
    ap.add_argument("--preflight", action="store_true",
                    help="first --cache x first --rung only (a small real run for "
                         "`vqlab queue --preflight`); not a result")
    ap.add_argument("--allow-build-mismatch", action="store_true",
                    help="score against caches built with different numerics (mlx, "
                         "mlx-lm, arch file, scorer variant). REFUSED by default; "
                         "recorded in every cell and in --out when allowed.")
    a = ap.parse_args()
    if a.preflight:
        a.cache, a.rung = a.cache[:1], a.rung[:1]
        print(f"PREFLIGHT: {a.cache[0]} x {a.rung[0]} only", flush=True)

    caches, rungs = _kv(a.cache, "cache"), _kv(a.rung, "rung")
    meta = {}
    for name, d in caches.items():
        m = json.load(open(os.path.join(d, "meta.json")))
        corpus = m["corpus"]
        if "/" not in corpus:                    # a house corpus NAME (prose / code / lit)
            corpus = str(_layout.corpus(corpus))
        elif not os.path.isabs(corpus):
            corpus = os.path.join(os.path.dirname(os.path.dirname(
                os.path.dirname(HERE))), corpus)     # repo root
        corpus = _layout.legacy_path(corpus)    # pre-split caches
        meta[name] = {"dir": d, "corpus": corpus,
                      "tokens": m.get("seq_len") or (m["tokens"] - 1),
                      "chunk": m.get("chunk", 512)}
        if not os.path.exists(corpus):
            raise SystemExit(f"FAIL: cache {name} names a corpus that is not "
                             f"here: {corpus}")

    # A pinned rung (vqlab pin) must have passed its smoke; checked for EVERY
    # rung before the first GPU minute is spent. Unpinned dirs pass unchanged.
    from vqlab.gate.pin import check_pin
    refused = []
    for rname, rdir in rungs.items():
        ok, msg = check_pin(rdir)
        if msg:
            print(f"[kl-ladder] {rname}: {msg}", flush=True)
        if not ok:
            refused.append(rname)
    if refused:
        raise SystemExit(f"FAIL: refusing to score unsmoked or changed pins: {', '.join(refused)}")

    # THE NUMERICS BUILD, checked before the first GPU minute: the scoring
    # interpreter's mlx / mlx-lm / scorer variant against every cache's stamp
    # (F194). stream_score re-checks each cell, arch file included, after its
    # (lazy) load; this catches the common case, a cache from the other env,
    # before any model is touched.
    run_build = _preflight_builds(a.python, rungs, meta, caches, a.allow_build_mismatch)

    # What each rung IS, stamped before the first GPU minute and re-checked
    # after the last: another session rebundling a rung mid-campaign (F151)
    # is a filesystem write no power gate can see.
    from vqlab.records.provenance import measured
    before = {r: measured(d) for r, d in rungs.items()}

    ppdir = a.per_pos_dir or os.path.join(
        os.path.dirname(a.out) if a.out else ".", "kl_per_position")
    os.makedirs(ppdir, exist_ok=True)

    table = {}
    for rname, rdir in rungs.items():
        table[rname] = {}
        for cname, cm in meta.items():
            print(f"[kl-ladder] {rname} x {cname}", flush=True)
            pp = os.path.join(ppdir, f"{rname}__{cname}.safetensors")
            # RESUME. Each rung x corpus record is saved beside its per-position
            # array the moment it finishes, and reused on a rerun if the cache,
            # model and array still match -- a crash late in a ladder no longer
            # throws away every rung scored before it (2026-09-24).
            rj = pp + ".json"
            rec = None
            if os.path.exists(pp) and os.path.exists(rj):
                old_rec = json.load(open(rj))
                ob = old_rec.get("numerics")
                same_build = not (ob and run_build) or all(
                    ob.get(k) == run_build.get(k) for k in PREFLIGHT_FIELDS)
                if (old_rec.get("model") == rdir
                        and old_rec.get("_cache_dir") == cm["dir"] and same_build):
                    rec = old_rec
                    print("    (resumed from saved record)", flush=True)
            if rec is None:
                rec = score_one(a.python, rdir, cm["dir"], cm["corpus"],
                                cm["tokens"], cm["chunk"], a.stream_ple,
                                a.lazy_over_gb, per_pos=pp,
                                allow_build_mismatch=a.allow_build_mismatch)
                rec["_cache_dir"] = cm["dir"]
                with open(rj + ".tmp", "w") as f:
                    json.dump(rec, f)
                os.replace(rj + ".tmp", rj)
            table[rname][cname] = rec
            print(f"    KL {rec['mean_kl_millinats']:.3f} "
                  f"+/- {rec.get('kl_sem_millinats', float('nan')):.3f} "
                  f"(ppl {rec['ppl']:.4f})", flush=True)

    ref = next(iter(rungs))
    names = list(caches)

    def paired(rname, cname):
        """Paired difference vs the reference on identical positions.

        Both rungs saw the SAME positions and the SAME teacher, so the
        per-position differences remove the position-to-position variance
        that dominates each rung's own SEM. Comparing two overlapping 95%
        intervals is NOT this test and is far more conservative -- an
        overlap there does not mean "no difference".
        """
        import numpy as np
        from safetensors.numpy import load_file
        fa = os.path.join(ppdir, f"{rname}__{cname}.safetensors")
        fb = os.path.join(ppdir, f"{ref}__{cname}.safetensors")
        if not (os.path.exists(fa) and os.path.exists(fb)):
            return None
        x = load_file(fa)["kl_millinats"].astype(np.float64)
        y = load_file(fb)["kl_millinats"].astype(np.float64)
        if x.shape != y.shape:
            return None
        d = x - y                                   # rung minus reference
        n = d.size
        sem = float(d.std(ddof=1) / np.sqrt(n))
        m = float(d.mean())
        return {"delta": m, "sem": sem, "t": (m / sem) if sem else 0.0, "n": n}

    print(f"\nKL to teacher, millinats at 12288 tokens (reference: {ref})")
    print("paired delta = this rung minus reference on identical positions; "
          "|t|>2 is a difference")
    print("rung".ljust(12) + "".join(f"{c:>34s}" for c in names))
    for rname in rungs:
        cells = []
        for c in names:
            r = table[rname][c]
            m, e = r["mean_kl_millinats"], r.get("kl_sem_millinats", 0.0)
            if rname == ref:
                cells.append(f"{m:14.2f}+/-{e:5.2f}{'':>13s}")
                continue
            pd = paired(rname, c)
            if pd is None:
                cells.append(f"{m:14.2f}+/-{e:5.2f}{'  (unpaired)':>13s}")
                continue
            tag = "SAME" if abs(pd["t"]) < 2 else (
                "BETTER" if pd["delta"] < 0 else "WORSE")
            cells.append(f"{m:14.2f}  d={pd['delta']:+7.2f} t={pd['t']:+6.1f} "
                         f"{tag:6s}")
        print(rname.ljust(12) + "".join(cells))

    after = {r: measured(d) for r, d in rungs.items()}
    changed = [r for r in rungs if (before[r].get("runtime"), before[r].get("fingerprint"))
               != (after[r].get("runtime"), after[r].get("fingerprint"))]
    if changed:
        print(f"\nWARNING: rung(s) CHANGED during this run (runtime or bytes): "
              f"{', '.join(changed)} -- their numbers describe no single artifact (F151)",
              flush=True)
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(
            {"measured": before, "changed_during_run": changed,
             "allow_build_mismatch": bool(a.allow_build_mismatch),
             "caches": meta, "rungs": rungs, "reference": ref,
             "table": table, "per_position_dir": ppdir,
             "paired": {r: {c: paired(r, c) for c in names}
                        for r in rungs if r != ref}}, indent=1))
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
