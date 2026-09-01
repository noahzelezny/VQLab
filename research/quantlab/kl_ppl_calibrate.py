#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Cross-calibrate KL-to-bf16 against perplexity, on a model where BOTH work.

WHY. `kl_damage.py` is the only quality instrument that survives on
gemma-4 (its raw-text likelihood is collapsed — GEMMA4_PPL_ANOMALY.md). But
KL is measured in millinats, and nobody has an intuition for what "8.4
millinats" costs in practice. Every ppl-based rule of thumb this lab has
accumulated ("+1% ppl is fine", "the knee is where ppl turns up") is stated
in units the gemma ladder cannot use.

Qwen is the bridge. Its perplexity is sane (referee ppl 2.35-3.17 across the
397B ladder; Qwen3.6-35B hellaswag 0.76 on our harness), so BOTH instruments
are valid on it simultaneously. Measure a Qwen ladder with both, fit the
relationship, and you can read a gemma KL number in ppl-equivalent terms.

WHAT IT REPORTS, per rung:
  - referee ppl on the wikitext corpus (the lab's instrument of record)
  - ppl ratio vs the bf16 teacher (the number the ladder actually cares
    about: "this rung costs +4% ppl")
  - mean KL to bf16 in millinats, and top-1 agreement
so the output table IS the conversion chart.

CAVEAT WORTH KEEPING. The mapping is model- and corpus-specific; it is a
sense-of-scale aid, not a law. A KL that costs +2% ppl on a dense Qwen need
not cost +2% on a gemma MoE. Use it to know whether a gemma KL is in
"barely moved" or "badly broken" territory, not to quote a ppl number for a
model whose ppl is meaningless.

    ./kl_ppl_calibrate.py --teacher <bf16 dir> --rungs <dir1> <dir2> ... \
        --cache-dir kl_cache_qwen --corpus referee/referee_corpus.txt
"""
import argparse
import json
import math
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).parent
PY = str(HERE / "venv" / "bin" / "python")


def ppl_of(model_dir, corpus, max_tokens):
    """Perplexity via a direct forward — same math as the referee, no
    layer streaming (these models fit; streaming is for the 397B)."""
    code = f'''
import mlx.core as mx, math
from mlx_lm.utils import load
m, t = load({str(model_dir)!r})
ids = t.encode(open({str(corpus)!r}).read())[:{max_tokens} + 1]
bos = getattr(t, "bos_token_id", None)
if bos is not None and (not ids or ids[0] != bos):
    ids = [bos] + ids[:{max_tokens}]
lg = m(mx.array([ids[:-1]])).astype(mx.float32)[0]
tgt = mx.array(ids[1:])
lse = mx.logsumexp(lg, axis=-1)
pk = mx.take_along_axis(lg, tgt[:, None].astype(mx.int64), axis=-1)[:, 0]
print("PPL", math.exp(float(mx.mean(lse - pk).item())))
'''
    out = subprocess.run([PY, "-c", code], capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("PPL"):
            return float(line.split()[1])
    print(out.stderr[-500:], file=sys.stderr)
    return None


def kl_of(model_dir, cache_dir):
    out = subprocess.run(
        [PY, str(HERE / "kl_damage.py"), "score",
         "--model", str(model_dir), "--cache-dir", str(cache_dir)],
        capture_output=True, text=True)
    try:
        blob = out.stdout[out.stdout.index("{"):out.stdout.rindex("}") + 1]
        r = json.loads(blob)
        return r["mean_kl_millinats"], r["top1_agreement"]
    except (ValueError, KeyError):
        print(out.stderr[-500:], file=sys.stderr)
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--rungs", nargs="+", required=True)
    ap.add_argument("--cache-dir", required=True,
                    help="teacher cache from `kl_damage.py cache`")
    ap.add_argument("--corpus", default="referee/referee_corpus.txt")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"teacher ppl ({args.corpus}, {args.max_tokens} tok)...", flush=True)
    base = ppl_of(args.teacher, args.corpus, args.max_tokens)
    print(f"  bf16 ppl {base:.4f}\n", flush=True)

    rows = []
    for r in args.rungs:
        name = pathlib.Path(r).name
        print(f"--- {name}", flush=True)
        p = ppl_of(r, args.corpus, args.max_tokens)
        kl, ag = kl_of(r, args.cache_dir)
        rows.append({"rung": name, "ppl": p,
                     "ppl_ratio": (p / base) if (p and base) else None,
                     "kl_millinats": kl, "top1_agreement": ag})
        print(f"  ppl {p}  KL {kl} millinats  agree {ag}\n", flush=True)

    print(f"\n{'rung':<10} {'ppl':>10} {'vs bf16':>9} "
          f"{'KL(mnats)':>11} {'top-1 agree':>12}")
    print(f"{'bf16':<10} {base:>10.4f} {'1.000x':>9} "
          f"{'0':>11} {'100.00%':>12}")
    for r in rows:
        if r["ppl"] is None or r["kl_millinats"] is None:
            print(f"{r['rung']:<10} {'FAILED':>10}")
            continue
        print(f"{r['rung']:<10} {r['ppl']:>10.4f} "
              f"{r['ppl_ratio']:>8.3f}x {r['kl_millinats']:>11.3f} "
              f"{r['top1_agreement']:>11.2%}")
    print("\nRead across: this is the KL -> ppl-cost conversion chart. "
          "Model/corpus specific — a sense-of-scale aid, not a law.")

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(
            {"teacher": args.teacher, "bf16_ppl": base,
             "corpus": args.corpus, "rungs": rows}, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
