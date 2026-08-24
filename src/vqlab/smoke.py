#!/usr/bin/env python3
"""vqlab smoke — generate one token through the runtime the artifact SHIPS.

This is the gate the rest of the pipeline cannot replace. Every byte-level
check reads bytes; none runs the model. An artifact in this project passed the
outlier gate, the release checks, the vision check AND the scoring referee,
then raised on its first real forward pass — because the referee scores
through the reference decode path while serving uses the fused kernel. A rung
can score normally and be unable to serve.

It also NAMES THE RUNTIME THAT ACTUALLY RESOLVED. Which copy executes — the
artifact's bundled model.py, site-packages, or a repo checkout — depends on
the loader and the environment, and assuming it has produced wrong
conclusions in both directions. So this prints `__file__` for the modules that
really loaded rather than asserting which one should have.

Generation needs the WHOLE model resident: this refuses to run an artifact
larger than the box unless you override, because a smoke that thrashes swap
produces no verdict.

    vqlab smoke <artifact> [--prompt TEXT] [--max-tokens N] [--headroom F]
                           [--skip-preflight]
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vqlab smoke",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("artifact")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-tokens", type=int, default=4,
                    help="tokens to generate. One is enough to satisfy the "
                         "rule; the default of 4 also catches a model that "
                         "produces exactly one token and then dies.")
    ap.add_argument("--headroom", type=float, default=0.90)
    ap.add_argument("--skip-preflight", action="store_true",
                    help="run even if the artifact may not fit in RAM. You "
                         "will get swap thrash and no verdict; this exists "
                         "for boxes whose memory the preflight misreads.")
    a = ap.parse_args(argv)
    art = pathlib.Path(a.artifact)
    if not (art / "config.json").exists():
        print(f"FAIL: {art} has no config.json")
        return 1

    if not a.skip_preflight:
        p = subprocess.run([sys.executable, str(HERE / "preflight_ram.py"),
                            str(art), "--headroom", str(a.headroom)])
        if p.returncode != 0:
            print("FAIL: artifact does not fit in RAM — a generation smoke "
                  "CANNOT produce a verdict here. Use a bigger box, or "
                  "--skip-preflight if you know better than the check.")
            return 1

    from mlx_lm.utils import load
    from mlx_lm import generate

    print(f"loading {art} ...", flush=True)
    model, tokenizer = load(str(art))

    # III.13: instrument the import, never assume which copy runs.
    print("\nRESOLVED RUNTIME (the copy that actually loaded):")
    seen, vq_modules = set(), 0
    for name, mod in model.named_modules() if hasattr(model, "named_modules") \
            else []:
        cls = type(mod)
        if cls.__name__.startswith("VQ"):
            vq_modules += 1
            origin = sys.modules.get(cls.__module__)
            path = getattr(origin, "__file__", "<unknown>")
            key = (cls.__name__, path)
            if key not in seen:
                seen.add(key)
                print(f"  {cls.__name__:18s} <- {path}")
    if not seen:
        print("  (no VQ modules found — is this a VQ artifact?)")
    print(f"  {vq_modules} VQ module(s) instantiated")

    print(f"\ngenerating {a.max_tokens} token(s) through the fused path ...",
          flush=True)
    try:
        text = generate(model, tokenizer, prompt=a.prompt,
                        max_tokens=a.max_tokens, verbose=False)
    except Exception as e:
        print(f"\nFAIL: the artifact could not generate: "
              f"{type(e).__name__}: {e}\n\n"
              "This is the failure every byte-level gate is blind to. Do NOT "
              "release this artifact; a score from it describes a model that "
              "cannot serve.")
        return 1
    if not text or not text.strip():
        print("\nFAIL: generation returned empty text.")
        return 1
    print(f"\n  {a.prompt!r} -> {text!r}")
    print("\nPASS: artifact generated through its shipping runtime. "
          "Record the resolved paths above in any runtime-dependent claim.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
