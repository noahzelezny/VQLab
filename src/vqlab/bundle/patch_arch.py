#!/usr/bin/env python3
"""Surgically replace a bundle's arch-resolution block. Nothing else moves.

For an artifact whose published runtime is far behind the repo (the 397B rungs
are ~324 code lines and 8 tracked flags behind), a full rebundle is a runtime
UPGRADE that cannot be text-gated on this hardware. This instead edits the
published model.py in place: the four-line mlx_lm-first `try/except` that
binds the base arch once at build time (F153) is replaced with the
loader-aware resolver (F159), and every other byte -- the runtime users
already have, the re-export loop, the config coercion, the output wrapper --
is left exactly as shipped. Under mlx_lm the resulting bundle binds the same
arch it always did, running the same runtime: the text path is byte-for-byte
the published behaviour. Only the mlx_vlm branch is new.

Refuses unless the block is found EXACTLY ONCE, and refuses to write a file
that does not compile.

    vqlab patch-arch <published-model.py> --out <patched-model.py>
"""
import argparse
import pathlib

OLD = '''try:
    _arch = _importlib.import_module(f"mlx_lm.models.{_cfg['model_type']}")
except ModuleNotFoundError:
    _arch = _importlib.import_module(f"mlx_vlm.models.{_cfg['model_type']}")
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = pathlib.Path(a.model_py).read_text()
    n = src.count(OLD)
    if n != 1:
        raise SystemExit(f"REFUSING: expected the mlx_lm-first block exactly once, "
                         f"found {n}. This is not a bundle this tool understands.")
    from vqlab.arch_resolve import PRELUDE
    # PRELUDE also carries the re-export loop and ModelArgs/ModelConfig lines,
    # which the published shim already has right after this block. Take only
    # the resolver: everything up to (not including) the re-export comment.
    marker = "# Re-export the base module's WHOLE public surface"
    resolver = PRELUDE[:PRELUDE.index(marker)].rstrip() + "\n"
    out = src.replace(OLD, resolver)
    compile(out, "model.py", "exec")
    pathlib.Path(a.out).write_text(out)
    added = out.count("\n") - src.count("\n")
    print(f"patched: replaced the 4-line resolver with {resolver.count(chr(10))} "
          f"lines (+{added} net); everything else byte-identical -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
