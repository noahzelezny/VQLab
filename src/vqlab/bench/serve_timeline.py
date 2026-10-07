#!/usr/bin/env python
"""serve-timeline — where a SERVED request's time goes, inside Knurlogic.

decode-timeline and prefill-timeline time a plain forward. A user never
runs a plain forward: their request goes through Knurlogic's HTTP handler,
template, prompt cache, memory admission, scheduler and executor, and F188
found a served prefill near 97 tok/s against ~245 for a plain forward of
the same bundle (at UNEQUAL prompt lengths -- unverified until re-measured
at equal length; that is the first thing to run this for).

Knurlogic partitions each request's wall time itself (engine/runtime/
spans.py; usage.knurlogic.timing.spans_s, on unless
KNURLOGIC_TIMING_SPANS=off) with a cursor, so the buckets sum to the whole
by construction. This tool drives n requests of a stated prompt length,
collects those partitions and prints the median of each bucket:

    http_build       request JSON -> a scheduler job (parse, translate)
    queue            submitted -> the scheduler started admitting it
    tokenize         chat template + tokenizer (vision: the image tower)
    admit_memory     fitting the prompt into memory (may evict cache)
    cache_fetch      prompt-cache lookup
    admit_other      the rest of admission (executor insert, request setup)
    prefill_gap      scheduler work between this request's prefill steps
    prefill_forward  the executor steps that computed its prompt
    prefill_host     after those steps: events, progress, first delta
    decode_*         the same three, per decode step, summed
    decode_forward_shared  decode steps that also prefilled ANOTHER prompt
    prefill_waiting  steps that admitted another request while this one
                     waited its turn (the executor admits one per step)

A served prefill slower than a plain forward is then a named bucket: if
prefill_forward alone is slow, the engine is (chunk size, kernel path);
if the time is in gap/host/admission, the serving layer is.

Each request gets a FRESH prompt (a different slice of the house corpus),
so the prompt cache cannot supply it, and a same-length warm-up is sent
first and discarded. n >= 3, prompt length stated, as Rule III asks.

    vqlab serve-timeline --url http://127.0.0.1:8080 --model <name> \
        --prompt-tokens 2048 [--gen-tokens 64] [--n 3] [--json-out f.json]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # src/
from vqlab.bench.speed_pair_knurlogic import _post, _prompt  # noqa: E402

TTFT_SIDE = ("http_build", "queue", "tokenize", "admit_memory", "cache_fetch",
             "admit_other", "prefill_waiting", "prefill_gap", "prefill_forward",
             "prefill_host")


def quiet_except_the_server() -> None:
    """Rule III, minus the one heavy process this tool exists to measure.
    box_quiet's guard would refuse here: a running `knurlogic serve` is on
    its heavy list, and it is the thing under test. Every OTHER kind of
    contention still refuses."""
    import os

    from vqlab.bench import box_quiet
    st = box_quiet.state()
    busy = [b for b in st["busy"] if "GPU-heavy" not in b]
    others = [h for h in st["heavy"] if "knurlogic" not in h]
    if others:
        busy.append(f"{len(others)} other GPU-heavy process(es): {others[0]}")
    print(f"[box] load1 {st['load1']} on {st['ncpu']} cores; busy: "
          f"{busy or 'no'}", flush=True)
    if busy and os.environ.get("VQLAB_ALLOW_BUSY") != "1":
        raise SystemExit("REFUSED: this Mac is busy with more than the server "
                         "under test:\n  - " + "\n  - ".join(busy) +
                         "\nWait, or set VQLAB_ALLOW_BUSY=1 and label the "
                         "result contended.")


def one(url, model, prompt_tokens, gen_tokens, offset):
    doc = {"model": model, "temperature": 0, "max_tokens": gen_tokens,
           "stream": False,
           "messages": [{"role": "user",
                         "content": _prompt(prompt_tokens, offset)}]}
    r = _post(url, doc)
    u = r.get("usage") or {}
    t = (u.get("knurlogic") or {}).get("timing") or {}
    if "spans_s" not in t:
        raise SystemExit(
            "the server sent no usage.knurlogic.timing.spans_s: it predates "
            "request spans, or runs with KNURLOGIC_TIMING_SPANS=off")
    return {"prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"),
            "cached_tokens": t.get("prompt_cached_tokens"),
            "ttft_s": t.get("ttft_s"), "prefill_tok_s": t.get("prefill_tok_s"),
            "decode_tok_s": t.get("decode_tok_s"),
            "chunk": t.get("prefill_chunk"),
            "spans": t["spans_s"], "whole_s": t["spans_whole_s"],
            "unaccounted_s": t["spans_unaccounted_s"]}


def summarize(runs: list[dict]) -> dict:
    keys = sorted({k for r in runs for k in r["spans"]})
    med = {k: statistics.median(r["spans"].get(k, 0.0) for r in runs)
           for k in keys}
    lo = {k: min(r["spans"].get(k, 0.0) for r in runs) for k in keys}
    hi = {k: max(r["spans"].get(k, 0.0) for r in runs) for k in keys}
    whole = statistics.median(r["whole_s"] for r in runs)
    ttft_side = sum(med.get(k, 0.0) for k in TTFT_SIDE)
    return {"median_s": med, "min_s": lo, "max_s": hi, "whole_s": whole,
            "ttft_side_s": ttft_side,
            "worst_unaccounted_s": max(abs(r["unaccounted_s"]) for r in runs)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", required=True, help="the Knurlogic server")
    ap.add_argument("--model", default="", help="served model name")
    ap.add_argument("--prompt-tokens", type=int, default=2048)
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    quiet_except_the_server()
    if a.n < 3:
        print("note: n < 3; Rule III asks for n >= 3 with scatter",
              file=sys.stderr)

    stride = a.prompt_tokens * 4 + 4096
    one(a.url, a.model, a.prompt_tokens, 2, 0)          # warm-up, discarded
    runs = [one(a.url, a.model, a.prompt_tokens, a.gen_tokens,
                stride * (i + 1)) for i in range(a.n)]
    for r in runs:
        if r["cached_tokens"]:
            print(f"WARNING: a timed request had {r['cached_tokens']} prompt "
                  "tokens from the prompt cache; its prefill is not a full "
                  "prefill", file=sys.stderr)
    s = summarize(runs)

    print(f"\nserve-timeline  {a.model or a.url}   prompt "
          f"{runs[0]['prompt_tokens']} tokens, {a.gen_tokens} generated, "
          f"n={a.n}, chunk {runs[0]['chunk']}")
    print(f"  ttft  {', '.join(str(r['ttft_s']) for r in runs)} s"
          f"   prefill tok/s {', '.join(str(r['prefill_tok_s']) for r in runs)}"
          f"   decode tok/s {', '.join(str(r['decode_tok_s']) for r in runs)}")
    print(f"  whole request (median) {s['whole_s']:.3f} s; partition check: "
          f"worst unaccounted {s['worst_unaccounted_s']*1e3:.2f} ms\n")
    print(f"  {'bucket':<24} {'median s':>9} {'min':>8} {'max':>8} {'% whole':>8}")
    for k, v in sorted(s["median_s"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:<24} {v:>9.4f} {s['min_s'][k]:>8.4f} "
              f"{s['max_s'][k]:>8.4f} {100*v/max(s['whole_s'], 1e-9):>7.1f}%")
    pf = s["median_s"].get("prefill_forward", 0.0)
    side = s["ttft_side_s"]
    if side > 0:
        print(f"\n  first-token side {side:.3f} s, of which the engine's own "
              f"prefill steps {pf:.3f} s ({100*pf/side:.0f}%). The rest "
              f"({side-pf:.3f} s) is serving: admission, scheduling, host "
              "work between steps.")
        n = runs[0]["prompt_tokens"] or a.prompt_tokens
        if pf > 0:
            print(f"  engine-only prefill rate: {n/pf:.0f} tok/s -- compare "
                  "THIS with a plain forward (prefill-timeline / decode-ladder "
                  "--mode prefill) at the same prompt length and chunk")
    print()
    if a.json_out:
        pathlib.Path(a.json_out).write_text(json.dumps(
            {"url": a.url, "model": a.model, "prompt_tokens": a.prompt_tokens,
             "gen_tokens": a.gen_tokens, "n": a.n, "runs": runs,
             "summary": s}, indent=1))
        print(f"  -> {a.json_out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
