#!/usr/bin/env python3
"""vqlab speed-pair-knurlogic: speed-pair for models served by Knurlogic.

Same rule as `vqlab speed-pair` (FINDINGS III): arms ALTERNATING (A B A B
...), one fresh load per arm per repetition, n >= 3, prompt length stated,
the answer is the per-pair RATIO B/A. The difference is WHO loads: Knurlogic
places the model -- on one Mac, or split across several (pipeline) -- which
is how anything too big for one box gets measured at all.

    vqlab speed-pair-knurlogic <name_a> <name_b> --machine "M4"
        [--machine "M3" --split pipeline --link tcp]
        [--prompt-tokens 2048] [--gen-tokens 128] [--n 3] [--out runs.jsonl]

Arms are Knurlogic MODEL NAMES (a symlink under the models dir of EVERY
machine used), never paths. Two builds that differ only in model.py share a
Knurlogic identity (it hashes config + shards), so they MUST be loaded by
distinct names; each record keeps the name Knurlogic reports back and its
runtime, and the tool REFUSES a result whose served name is not the arm or
whose runtime is not "bundled" (knurlogic's own vendored runtime would not
be the code under test).

Timing is Knurlogic's own: usage.knurlogic.timing {prefill_tok_s,
decode_tok_s} of one greedy /v1/chat/completions request, after an 8-token
warm-up request on the same load. Requires the Knurlogic source tree
(--knurlogic-src) and its page running on this Mac.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # src/
from vqlab import _layout  # noqa: E402

KSRC = "~/Documents/AgenicAI/knurlogic-cluster/src"


def _prompt(prompt_tokens):
    # ~4 chars/token on the house prose corpus; the served prompt_tokens
    # count comes back in the response and is what gets recorded.
    text = _layout.corpus("prose").read_text()
    return text[: prompt_tokens * 4]


def _post(url, doc, timeout=1800):
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(doc).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _vq_runtime(url):
    """"bundled" / "knurlogic" from the server's /status.json (state.SERVED
    ["runtime"]), or None if it does not say. The state row's own `runtime`
    field is the SERVER kind, not which VQ runtime loaded."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/status.json", timeout=30) as r:
            doc = json.loads(r.read())
    except Exception:
        return None
    stack = [doc]
    while stack:
        d = stack.pop()
        if isinstance(d, dict):
            v = d.get("runtime")
            if v in ("bundled", "knurlogic"):
                return v
            stack.extend(d.values())
        elif isinstance(d, list):
            stack.extend(d)
    return None


def _ready(e):
    return e.get("phase") == "ready" or (not e.get("job") and e.get("state") == "loaded")


def _entry(K, job):
    for m in K.state().get("models", []):
        if str(m.get("job")) == str(job) or str(m.get("instance")) == str(job):
            return m
    return None


def run_arm(K, name, a, prompt):
    out = K.load(artifact=name, machines=a.machine, split=a.split, link=a.link)
    job = out.get("job") or out.get("instance")
    if not job or out.get("refused") or out.get("error"):
        raise SystemExit(f"FAIL: load {name}: {out}")
    t0 = time.time()
    try:
        while True:
            e = _entry(K, job)
            if e and _ready(e):
                break
            if e and (e.get("phase") in ("failed", "stopped") or e.get("state") == "failed"):
                raise SystemExit(f"FAIL: {name} job {job} phase {e.get('phase')}: {e}")
            if time.time() - t0 > a.load_timeout:
                raise SystemExit(f"FAIL: {name} job {job} not ready after {a.load_timeout}s")
            time.sleep(10)
        load_s = time.time() - t0
        served = e.get("name")
        url = out.get("url") or e.get("url") or e.get("where")
        runtime = _vq_runtime(url)
        base = {"model": name, "temperature": 0,
                "messages": [{"role": "user", "content": prompt}]}
        _post(url, dict(base, max_tokens=8))                      # warm-up
        r = _post(url, dict(base, max_tokens=a.gen_tokens))
        u = r.get("usage", {})
        t = (u.get("knurlogic") or {}).get("timing") or {}
        rec = {"arm": name, "served_name": served, "runtime": runtime,
               "job": job, "machines": a.machine, "split": a.split, "link": a.link,
               "load_s": round(load_s, 1), "prompt_tokens": u.get("prompt_tokens"),
               "gen_tokens": u.get("completion_tokens"),
               "prefill_tok_s": t.get("prefill_tok_s"), "decode_tok_s": t.get("decode_tok_s"),
               "ttft_s": t.get("ttft_s")}
    finally:
        K.unload(job=str(job))
        for _ in range(60):                       # wait until it is gone everywhere
            if _entry(K, job) is None:
                break
            time.sleep(5)
    bad = []
    if served and served != name:
        bad.append(f"served as {served!r}, not {name!r} (identity collapse?)")
    if runtime == "knurlogic":
        bad.append("served on knurlogic's vendored runtime, not the bundled model.py")
    elif runtime is None:
        print(f"WARNING {name}: /status.json did not say which VQ runtime loaded; "
              "check the rank logs for 'ships its own runtime'", file=sys.stderr)
    if not rec["decode_tok_s"]:
        bad.append("no usage.knurlogic.timing in the response")
    if bad:
        raise SystemExit(f"FAIL: {name}: " + "; ".join(bad) + f"\n{json.dumps(rec)}")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vqlab speed-pair-knurlogic",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("arm_a")
    ap.add_argument("arm_b")
    ap.add_argument("--machine", action="append", required=True,
                    help="Knurlogic machine name (repeat for a split)")
    ap.add_argument("--split", default="")
    ap.add_argument("--link", default="")
    ap.add_argument("--prompt-tokens", type=int, default=2048)
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--load-timeout", type=int, default=1800)
    ap.add_argument("--knurlogic-src", default=KSRC)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    sys.path.insert(0, a.knurlogic_src)
    from knurlogic.interfaces import mcp as K

    prompt = _prompt(a.prompt_tokens)
    recs = {a.arm_a: [], a.arm_b: []}
    for rep in range(1, a.n + 1):
        for arm in (a.arm_a, a.arm_b):
            rec = dict(run_arm(K, arm, a, prompt), rep=rep, n=a.n)
            print(json.dumps(rec), flush=True)
            if a.out:
                with open(a.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            recs[arm].append(rec)
    print(f"\nspeed-pair-knurlogic  B/A  (A = {a.arm_a}\n"
          f"                            B = {a.arm_b})\n"
          f"machines {a.machine} split={a.split or '-'} link={a.link or '-'}, "
          f"prompt {recs[a.arm_a][0]['prompt_tokens']} tokens, {a.gen_tokens} generated, "
          f"n={a.n}, alternating")
    for key, label in (("decode_tok_s", "decode"), ("prefill_tok_s", "prefill")):
        r = [b[key] / x[key] for x, b in zip(recs[a.arm_a], recs[a.arm_b])]
        print(f"  {label:8} ratio per pair {', '.join(f'{v:.3f}' for v in r)}   "
              f"median {statistics.median(r):.3f}  range {min(r):.3f}-{max(r):.3f}")
    if a.n < 3:
        print("  NOTE: n < 3 -- a smoke of the instrument, not a quotable ratio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
