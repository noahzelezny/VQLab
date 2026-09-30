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
decode_tok_s} of one greedy /v1/chat/completions request (MTP off by default), after a SAME-LENGTH
warm-up request on the same load (discarded). Which VQ runtime loaded is read from every rank LOG ("ships its own runtime ... WILL be executed"); nothing over HTTP carries it. Requires the Knurlogic source tree
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
    base = url.rstrip("/")
    if not base.endswith("/v1"):          # Knurlogic's load() url already ends in /v1
        base += "/v1"
    req = urllib.request.Request(base + "/chat/completions",
                                 data=json.dumps(doc).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


BUNDLED = "ships its own runtime (model.py) and it WILL be executed"
VENDORED = "runtime serves it instead"


def _rank_logs(job, hosts):
    """{machine: log text} for every rank of `job`. The rank log is the only
    authoritative record of which VQ runtime loaded (Knurlogic, 2026-09-29:
    SERVED["runtime"] is not exposed over HTTP; state()'s `runtime` is the
    server kind). `hosts` maps a machine name to "user@host" for ssh; the
    rest are read locally."""
    import glob
    import subprocess
    out = {}
    for machine, host in hosts.items():
        cmd = f"cat ~/.cache/knurlogic/jobs/{job}/rank*.log 2>/dev/null"
        if host:
            p = subprocess.run(["ssh", host, cmd], capture_output=True, text=True, timeout=60)
            out[machine] = p.stdout
        else:
            out[machine] = "".join(open(f).read() for f in
                                   glob.glob(str(pathlib.Path.home() /
                                                 f".cache/knurlogic/jobs/{job}/rank*.log")))
    return out


def _ready(e):
    return e.get("phase") == "ready" or (not e.get("job") and e.get("state") == "loaded")


def _entry(K, job):
    for m in K.state().get("models", []):
        if str(m.get("job")) == str(job) or str(m.get("instance")) == str(job):
            return m
    return None


def run_arm(K, name, a, prompt):
    out = K.load(artifact=name, machines=a.machine, split=a.split, link=a.link,
                 sets=a.sets or None)
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
        logs = _rank_logs(job, a.hosts)
        runtime = {m: ("bundled" if BUNDLED in t else "knurlogic" if VENDORED in t else None)
                   for m, t in logs.items()}
        named = {m: (f"artifact  {name}" in t) for m, t in logs.items()}
        base = {"model": name, "temperature": 0,
                "messages": [{"role": "user", "content": prompt}]}
        _post(url, dict(base, max_tokens=8))    # warm-up: SAME prompt length (new shapes compile), discarded
        r = _post(url, dict(base, max_tokens=a.gen_tokens))
        u = r.get("usage", {})
        t = (u.get("knurlogic") or {}).get("timing") or {}
        rec = {"arm": name, "served_name": served, "runtime_by_rank": runtime,
               "named_by_rank": named, "sets": a.sets,
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
    for m, rt in runtime.items():
        if rt != "bundled":
            bad.append(f"rank on {m}: runtime {rt!r} (log lacks {BUNDLED!r})")
        if not named[m]:
            bad.append(f"rank on {m}: log does not name artifact {name!r} (identity collapse?)")
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
    ap.add_argument("--set", action="append", default=[], metavar="K=V",
                    help="Knurlogic launch setting (repeatable); KNURLOGIC_MTP=off is the default "
                         "so decode speed reflects the kernels, not draft acceptance")
    ap.add_argument("--host", action="append", default=["M4=noahzelezny@M4"],
                    metavar="MACHINE=user@host", help="ssh target for a remote rank's logs")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    a.sets = {"KNURLOGIC_MTP": "off"}
    for kv in a.set:
        k, _, v = kv.partition("=")
        a.sets[k] = v
    remote = dict(h.split("=", 1) for h in a.host)
    a.hosts = {m: remote.get(m) for m in a.machine}
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
