#!/usr/bin/env python3
"""vqlab speed-pair: decode and prefill speed of two ARTIFACTS, as a ratio.

The lab's speed rule (FINDINGS III, bench/CONTEXT.md) as a tool: one fresh
process per arm per repetition, arms ALTERNATING (A B A B ...), n >= 3, prompt
length stated, and the answer is the per-pair RATIO B/A -- never an absolute,
because at large sizes the decode instrument is bimodal. prefill-bench
compares two codebook arms INSIDE one artifact; this compares two artifacts
(e.g. VQ vs affine at matched bytes).

    vqlab speed-pair <arm_a> <arm_b> [--prompt-tokens 2048] [--gen-tokens 128]
                     [--n 3] [--corpus prose] [--out runs.jsonl]

Each child loads the arm through the runtime it ships (mlx-lm load,
trust_remote_code), warms up on 64 tokens, then generates greedily from the
first --prompt-tokens tokens of the house corpus. Every child prints one JSON
line; the parent keeps them all (--out appends) and prints the ratios.

Refuses an arm that preflight_ram says cannot be resident, and an arm whose
vqlab_pin.json refuses it (gate/pin.check_pin). It cannot see a contended box:
run it under the GPU lease (vqlab queue does) with nothing else placed.
Paid for on night 4 (2026-09-26): the scratch version hardcoded the corpus
path, died on the restructure, and its queue logged "0 runs" as done.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # src/
from vqlab import _layout  # noqa: E402


def child(path, prompt_tokens, gen_tokens, corpus):
    import inspect
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    if "trust_remote_code" in inspect.signature(load).parameters:
        model, tok = load(path, trust_remote_code=True)
    else:
        model, tok = load(path, model_config={"trust_remote_code": True})
    ids = tok.encode(_layout.corpus(corpus).read_text())[:prompt_tokens]
    if len(ids) < prompt_tokens:
        raise SystemExit(f"FAIL: corpus {corpus!r} has only {len(ids)} tokens")
    greedy = make_sampler(temp=0.0)
    for _ in stream_generate(model, tok, prompt=tok.decode(ids[:64]), max_tokens=4, sampler=greedy):
        pass                                                   # warm-up
    last = None
    for r in stream_generate(model, tok, prompt=tok.decode(ids), max_tokens=gen_tokens, sampler=greedy):
        last = r
    if last is None or last.generation_tokens == 0:
        raise SystemExit("FAIL: generation returned nothing")
    print(json.dumps({"arm": str(path), "prompt_tokens": last.prompt_tokens,
                      "prompt_tps": round(last.prompt_tps, 2), "gen_tokens": last.generation_tokens,
                      "gen_tps": round(last.generation_tps, 2), "peak_gb": round(last.peak_memory, 2)}),
          flush=True)


def _gate(arm, headroom):
    from vqlab.gate.pin import check_pin
    ok, msg = check_pin(arm)
    if msg:
        print(msg)
    if not ok:
        raise SystemExit(f"FAIL: {arm} refused by its pin")
    p = subprocess.run([sys.executable, str(_layout.find("preflight_ram.py")), str(arm),
                        "--headroom", str(headroom)], capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"FAIL: {arm} cannot be resident here, a speed run would thrash:\n{p.stdout}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vqlab speed-pair", description=__doc__.split("\n")[0])
    ap.add_argument("arm_a")
    ap.add_argument("arm_b")
    ap.add_argument("--prompt-tokens", type=int, default=2048)
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--n", type=int, default=3, help="repetitions per arm (>= 3 to quote)")
    ap.add_argument("--corpus", default="prose")
    ap.add_argument("--headroom", type=float, default=0.90)
    ap.add_argument("--out", help="append every child's JSON line here")
    ap.add_argument("--_child", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a._child:
        child(a._child, a.prompt_tokens, a.gen_tokens, a.corpus)
        return 0

    for arm in (a.arm_a, a.arm_b):
        _gate(arm, a.headroom)
    rows = {a.arm_a: [], a.arm_b: []}
    for i in range(a.n):
        for arm in (a.arm_a, a.arm_b):
            p = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve()),
                                a.arm_a, a.arm_b, "--_child", arm,
                                "--prompt-tokens", str(a.prompt_tokens),
                                "--gen-tokens", str(a.gen_tokens), "--corpus", a.corpus],
                               capture_output=True, text=True)
            line = next((ln for ln in p.stdout.splitlines() if ln.startswith("{")), None)
            if p.returncode != 0 or line is None:
                sys.stderr.write(p.stderr[-3000:])
                raise SystemExit(f"FAIL: arm {arm} rep {i + 1} rc={p.returncode}, no result")
            rec = {**json.loads(line), "rep": i + 1, "n": a.n}
            rows[arm].append(rec)
            print(json.dumps(rec), flush=True)
            if a.out:
                with open(a.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")

    A, B = rows[a.arm_a], rows[a.arm_b]
    print(f"\nspeed-pair  B/A  (A = {a.arm_a}\n             B = {a.arm_b})")
    print(f"prompt {a.prompt_tokens} tokens ({a.corpus}), {a.gen_tokens} generated, n={a.n}, alternating")
    for key, label in (("gen_tps", "decode"), ("prompt_tps", "prefill")):
        r = [b[key] / x[key] for x, b in zip(A, B)]
        sp = f"  range {min(r):.3f}-{max(r):.3f}" if len(r) > 1 else ""
        print(f"  {label:8} ratio per pair {', '.join(f'{v:.3f}' for v in r)}   median {statistics.median(r):.3f}{sp}")
    print(f"  peak GB  A {max(x['peak_gb'] for x in A):.2f}   B {max(x['peak_gb'] for x in B):.2f}")
    if a.n < 3:
        print("  NOTE: n < 3 -- a smoke of the instrument, not a quotable ratio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
