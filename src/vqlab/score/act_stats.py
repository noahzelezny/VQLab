#!/usr/bin/env python
"""act-stats — how often does each clamp or limit in a family's PARITY fire
on the house corpora?

F195: DeepSeek's reference clamps the shared expert's SwiGLU at 10; mlx-lm
did not. Whether such a divergence matters is a RATE question: a clamp that
never fires on the corpora cannot move a KL number, one that fires on 0.03%
of activations can. Promoted from lab/scripts/shared_clip_rate_ds4.py.

Generic here: the corpus loop (prose / code / lit, BOS handling, token cap),
the clamp counter, the JSON lines. Family-specific: WHERE the clamped
activations live, which is the plugin's `act_stats` hook (family/<name>.py);
a family without one is refused. GPU: streams the teacher through the
family's validated scorer, one corpus at a time.

    vqlab act-stats deepseek_v4 --model <teacher> --tokens 12288
"""
from __future__ import annotations

import argparse
import json
import sys

from vqlab.core.source.clamps import ClampCounter, corpus_ids  # noqa: F401  (re-export)

CORPORA = ("prose", "code", "lit")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vqlab act-stats", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("family", help="family plugin name or model_type (e.g. deepseek_v4)")
    ap.add_argument("--model", required=True, help="teacher / artifact directory")
    ap.add_argument("--tokens", type=int, default=12288)
    ap.add_argument("--corpus", action="append", choices=CORPORA,
                    help="repeatable; default all three")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--lazy-over-gb", type=float, default=16.0)
    a = ap.parse_args(argv)

    from vqlab.family import get
    p = get(a.family)
    if p is None:
        sys.exit(f"FAIL: no family plugin {a.family!r}")
    if p.act_stats is None:
        sys.exit(f"FAIL: family {p.name} has no act_stats hook; add one to "
                 f"vqlab/family/{p.name}.py (see deepseek_v4 for the shape)")
    probes = sorted({x.probe for x in p.parity if x.probe})
    print(f"# act-stats {p.name}: probes {', '.join(probes) or '(none declared)'}", flush=True)
    for row in p.act_stats(a.model, a.corpus or list(CORPORA), a.tokens, a):
        print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
