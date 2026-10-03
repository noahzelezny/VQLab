#!/usr/bin/env python3
"""loo-bands: leave-one-out band hybrids of a VQ build, plus the KL queue.

    vqlab loo-bands --vq <VQ build> --exact <exact base> --bands 0-11,12-23,...
                    --out-dir <dir> [--cache prose=... --cache code=... --cache lit=...]

Each hybrid is the VQ build with ONE band restored to the exact base's
experts (symlinks, via `vqlab mix`: config maps follow the bytes). Scoring
every hybrid against the VQ build on the same teacher caches gives each
band's share of the damage measured IN the assembled network (compounding,
not isolation: AGENTS.md "isolation probes are anti-signal"). Bands that
remove the most KL when restored are the ones that deserve more bits.

With --cache, writes <out-dir>/loo-queue.json: one kl-ladder step with the
VQ build first (the paired reference), one rung per hybrid, per-position
arrays saved (re-pair later with kl-pair at zero GPU cost). Run it with
`vqlab queue run <file> --preflight`, then `--detach`.

This is how DeepSeek-V4-Flash's damage map (peak L24-35) was measured.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from vqlab.assemble import mix


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vqlab loo-bands", description=__doc__.split("\n")[0])
    ap.add_argument("--vq", required=True, help="the VQ build to probe")
    ap.add_argument("--exact", required=True, help="the exact base (same skeleton) to restore from")
    ap.add_argument("--bands", required=True, help="comma list of LO-HI")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache", action="append", default=[], help="name=teacher cache dir (kl-ladder)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--lazy-over-gb", default="16")
    a = ap.parse_args(argv)
    vq, exact, od = pathlib.Path(a.vq), pathlib.Path(a.exact), pathlib.Path(a.out_dir)
    bands = [tuple(int(x) for x in b.split("-")) for b in a.bands.split(",")]
    od.mkdir(parents=True, exist_ok=True)
    rungs = []
    for lo, hi in bands:
        out = od / f"loo-L{lo}-{hi}"
        if out.exists():
            print(f"{out.name} exists, kept")
        else:
            mix.build(out, vq, [(exact, lo, hi)])
        rungs.append((f"loo{lo}-{hi}", out))
    if a.cache:
        args = []
        for c in a.cache:
            args += ["--cache", c]
        args += ["--rung", f"vq={vq}"]
        for n, p in rungs:
            args += ["--rung", f"{n}={p}"]
        args += ["--python", a.python, "--lazy-over-gb", a.lazy_over_gb,
                 "--per-pos-dir", str(od / "per_position"), "--out", str(od / "kl_ladder_loo.json")]
        q = {"name": f"loo-{vq.name}", "commit": "HEAD", "python": a.python,
             "steps": [{"name": "kl-loo-bands", "cmd": "kl-ladder", "args": args,
                        "preflight": {"append": ["--preflight"]}, "retries": 1,
                        "timeout_s": 3600 * (1 + len(rungs))}]}
        (od / "loo-queue.json").write_text(json.dumps(q, indent=1))
        print(f"queue: {od / 'loo-queue.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
