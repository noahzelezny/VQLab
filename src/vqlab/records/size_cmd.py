#!/usr/bin/env python3
"""vqlab size: an artifact's size three ways, from its own headers.

    vqlab size <artifact> [--json]

text (the headline every card quotes), + vision tower, + MTP head (indexed
or sidecar), and the full download. The 397B indexes its MTP head while
GLM and DeepSeek ship it as a sidecar, so "index total" is not "text".
"""
from __future__ import annotations

import argparse
import json
import sys

from vqlab.core.artifact import sizes

GIB = 2 ** 30


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vqlab size", description=__doc__.split("\n")[0])
    ap.add_argument("artifact")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    s = sizes(a.artifact)
    if a.json:
        print(json.dumps(s, indent=1))
        return 0
    mtp = s["mtp"] + s["mtp_sidecar"]
    print(f"text weights      {s['text'] / GIB:8.2f} GiB")
    print(f"+ vision tower    {(s['text'] + s['tower']) / GIB:8.2f} GiB   (tower {s['tower'] / GIB:.2f})")
    print(f"+ MTP head        {(s['text'] + s['tower'] + mtp) / GIB:8.2f} GiB   (head {mtp / GIB:.2f}"
          + (f", sidecar {', '.join(s['mtp_files'])}" if s["mtp_files"] else ", indexed" if s["mtp"] else "")
          + ")")
    print(f"full download     {s['download'] / GIB:8.2f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
