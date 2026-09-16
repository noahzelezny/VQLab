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


def _kv(pairs, what):
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"FAIL: --{what} wants name=path, got {p!r}")
        k, v = p.split("=", 1)
        out[k] = v
    return out


def score_one(python, model, cache_dir, corpus, tokens, chunk, stream_ple,
              lazy_over_gb, per_pos=None):
    cmd = [python, "-m", "vqlab.stream_score", "--model", model,
           "--corpus", corpus, "--tokens", str(tokens), "--chunk", str(chunk),
           "--kl-cache", cache_dir, "--lazy-over-gb", str(lazy_over_gb)]
    if per_pos:
        cmd += ["--kl-per-position", per_pos]
    if stream_ple:
        cmd.append("--stream-ple")
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
    a = ap.parse_args()

    caches, rungs = _kv(a.cache, "cache"), _kv(a.rung, "rung")
    meta = {}
    for name, d in caches.items():
        m = json.load(open(os.path.join(d, "meta.json")))
        corpus = m["corpus"]
        if not os.path.isabs(corpus):
            corpus = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                                  corpus)
        meta[name] = {"dir": d, "corpus": corpus,
                      "tokens": m.get("seq_len") or (m["tokens"] - 1),
                      "chunk": m.get("chunk", 512)}
        if not os.path.exists(corpus):
            raise SystemExit(f"FAIL: cache {name} names a corpus that is not "
                             f"here: {corpus}")

    ppdir = a.per_pos_dir or os.path.join(
        os.path.dirname(a.out) if a.out else ".", "kl_per_position")
    os.makedirs(ppdir, exist_ok=True)

    table = {}
    for rname, rdir in rungs.items():
        table[rname] = {}
        for cname, cm in meta.items():
            print(f"[kl-ladder] {rname} x {cname}", flush=True)
            pp = os.path.join(ppdir, f"{rname}__{cname}.safetensors")
            rec = score_one(a.python, rdir, cm["dir"], cm["corpus"],
                            cm["tokens"], cm["chunk"], a.stream_ple,
                            a.lazy_over_gb, per_pos=pp)
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

    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(
            {"caches": meta, "rungs": rungs, "reference": ref,
             "table": table, "per_position_dir": ppdir,
             "paired": {r: {c: paired(r, c) for c in names}
                        for r in rungs if r != ref}}, indent=1))
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
