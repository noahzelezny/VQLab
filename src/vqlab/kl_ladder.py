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
              lazy_over_gb):
    cmd = [python, "-m", "vqlab.stream_score", "--model", model,
           "--corpus", corpus, "--tokens", str(tokens), "--chunk", str(chunk),
           "--kl-cache", cache_dir, "--lazy-over-gb", str(lazy_over_gb)]
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

    table = {}
    for rname, rdir in rungs.items():
        table[rname] = {}
        for cname, cm in meta.items():
            print(f"[kl-ladder] {rname} x {cname}", flush=True)
            rec = score_one(a.python, rdir, cm["dir"], cm["corpus"],
                            cm["tokens"], cm["chunk"], a.stream_ple,
                            a.lazy_over_gb)
            table[rname][cname] = rec
            print(f"    KL {rec['mean_kl_millinats']:.3f} "
                  f"+/- {rec.get('kl_sem_millinats', float('nan')):.3f} "
                  f"(ppl {rec['ppl']:.4f})", flush=True)

    ref = next(iter(rungs))
    names = list(caches)
    print(f"\nKL to teacher, millinats (reference: {ref})")
    head = "rung".ljust(14) + "".join(f"{c:>26s}" for c in names)
    print(head)
    for rname in rungs:
        cells = []
        for c in names:
            r = table[rname][c]
            m, e = r["mean_kl_millinats"], r.get("kl_sem_millinats", 0.0)
            v = ""
            if rname != ref:
                b = table[ref][c]
                bm, be = b["mean_kl_millinats"], b.get("kl_sem_millinats", 0.0)
                lo, hi = m - 1.96 * e, m + 1.96 * e
                blo, bhi = bm - 1.96 * be, bm + 1.96 * be
                v = "  SAME" if (lo <= bhi and blo <= hi) else \
                    ("  BETTER" if m < bm else "  WORSE")
            cells.append(f"{m:12.3f}+/-{e:6.3f}{v:>7s}")
        print(rname.ljust(14) + "".join(cells))

    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(
            {"caches": meta, "rungs": rungs, "reference": ref,
             "table": table}, indent=1))
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
