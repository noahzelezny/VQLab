#!/usr/bin/env python3
"""vqlab runtime-equiv — the same model slice through two interpreters: bitwise or not?

F194: scoring (mlx 0.32.0.dev + a grafted mlx-lm 0.31.9) and serving (mlx
0.31.2) disagreed by up to 1.0 logit on DeepSeek with the architecture code
bitwise identical; nobody would have known without a forward through each.
Run this whenever either environment moves. Promoted from
lab/scripts/runtime_equiv_ds4.py and generalized to any mlx-lm family.

    vqlab runtime-equiv --model <slice> --python-a <scoring python> \\
        --python-b <serving python> [--tokens 2048] [--chunk 512] [--knurlogic b]

Each side runs ONE chunked forward in its own interpreter (a worker process:
`python -m vqlab.score.runtime_equiv --worker ...`), over the same token ids
from the house prose corpus, and saves float32 logits plus its numerics build
(core.numerics). The parent compares them: BITWISE equal, or the max |logit
difference| and the argmax agreement. Use a small slice (`vqlab slice` or a
4-layer teacher cut): the point is the build, not the model.

--knurlogic SIDE loads that side's architecture from Knurlogic's vendored
module (serving's path) by calling its register() before the load.

Exit code: 0 bitwise equal, 1 different, 2 a side failed.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np


# ------------------------------------------------------------------ worker
def worker(model, out, tokens, chunk, knurlogic=False) -> int:
    """One chunked forward in THIS interpreter; logits -> out (.npy), build -> stdout."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # src/
    from vqlab import _layout
    from vqlab.core import numerics
    mp = pathlib.Path(model)
    mt = numerics.model_type_of(mp)
    if knurlogic:
        from knurlogic.engine import register as kreg
        kreg.register(mt)
    import mlx.core as mx
    from mlx_lm.utils import load_model, load_tokenizer
    tok = load_tokenizer(mp)
    ids = tok.encode(pathlib.Path(_layout.corpus("prose")).read_text())[:tokens]
    m, _ = load_model(mp, lazy=False)
    if hasattr(m, "make_cache"):
        cache = m.make_cache()
    else:
        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(m)
    x = mx.array([ids])
    lg = []
    for s in range(0, len(ids), chunk):
        lg.append(m(x[:, s:s + chunk], cache=cache).astype(mx.float32)[0])
        mx.eval(lg[-1])
    np.save(out, np.array(mx.concatenate(lg, 0)))
    print(json.dumps({"tokens": len(ids), "chunk": chunk, "knurlogic": knurlogic,
                      "ids_head": ids[:8], "numerics": numerics.build(mt, model=m)}))
    return 0


# ------------------------------------------------------------------ compare
def compare(a: np.ndarray, b: np.ndarray) -> dict:
    """Bitwise first (float32 bit patterns: catches -0.0 vs 0.0 and NaN
    payloads a value compare would miss), then the size of the difference."""
    if a.shape != b.shape:
        return {"bitwise": False, "shape_a": list(a.shape), "shape_b": list(b.shape),
                "note": "different shapes: the sides did not run the same slice"}
    a32, b32 = a.astype(np.float32), b.astype(np.float32)
    bitwise = bool(np.array_equal(a32.view(np.uint32), b32.view(np.uint32)))
    d = np.abs(a32.astype(np.float64) - b32.astype(np.float64))
    pos_max = d.max(axis=-1) if d.ndim > 1 else d
    return {"bitwise": bitwise,
            "max_abs_logit_diff": float(d.max()) if d.size else 0.0,
            "mean_abs_logit_diff": float(d.mean()) if d.size else 0.0,
            "positions_differing": int((pos_max > 0).sum()),
            "positions": int(pos_max.size),
            "argmax_agreement": float((a32.argmax(-1) == b32.argmax(-1)).mean())
            if a32.ndim > 1 else None}


def _run_side(python, model, out, tokens, chunk, knurlogic):
    cmd = [python, "-m", "vqlab.score.runtime_equiv", "--worker", "--model", model,
           "--out", out, "--tokens", str(tokens), "--chunk", str(chunk)]
    if knurlogic:
        cmd.append("--knurlogic-worker")
    r = subprocess.run(cmd, capture_output=True, text=True)
    line = [x for x in r.stdout.splitlines() if x.startswith("{")]
    if r.returncode != 0 or not line:
        tail = (r.stderr or r.stdout or "").strip().splitlines()
        return None, tail[-1] if tail else f"rc={r.returncode}, no output"
    return json.loads(line[-1]), None


def report(info_a, info_b, res) -> str:
    from vqlab.core.numerics import describe, diff
    na, nb = info_a.get("numerics"), info_b.get("numerics")
    lines = [f"  a: {describe(na)}  arch {str((na or {}).get('arch_sha256'))[:12]}"
             f"{'  (knurlogic)' if info_a.get('knurlogic') else ''}",
             f"  b: {describe(nb)}  arch {str((nb or {}).get('arch_sha256'))[:12]}"
             f"{'  (knurlogic)' if info_b.get('knurlogic') else ''}"]
    dd = diff(na, nb)
    lines.append("  builds differ in: " + ", ".join(k for k, _, _ in dd) if dd
                 else "  builds identical in every compared field")
    if info_a.get("ids_head") != info_b.get("ids_head"):
        lines.append("  WARNING: the sides tokenized differently; the logits are not comparable")
    if res["bitwise"]:
        lines.append(f"  BITWISE EQUAL over {res.get('positions')} positions")
    elif "max_abs_logit_diff" in res:
        lines.append(f"  DIFFERENT: max |d logit| {res['max_abs_logit_diff']:.6g}, mean "
                     f"{res['mean_abs_logit_diff']:.3g}, {res['positions_differing']}/"
                     f"{res['positions']} positions differ, argmax agreement "
                     f"{res['argmax_agreement']:.4%}")
    else:
        lines.append(f"  DIFFERENT: {res.get('note')}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vqlab runtime-equiv", description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="model dir (a small slice)")
    ap.add_argument("--python-a", default=sys.executable, help="interpreter A (default: this one)")
    ap.add_argument("--python-b", help="interpreter B")
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--knurlogic", action="append", choices=("a", "b"), default=[],
                    help="load this side's architecture from Knurlogic (repeatable)")
    ap.add_argument("--out", help="write the comparison JSON here")
    ap.add_argument("--keep-dir", help="keep both sides' logits (.npy) here")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--knurlogic-worker", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.worker:
        if not a.out:
            ap.error("--worker needs --out")
        return worker(a.model, a.out, a.tokens, a.chunk, a.knurlogic_worker)
    if not a.python_b:
        ap.error("--python-b is required (the second interpreter)")
    if a.python_a == a.python_b and set(a.knurlogic) in (set(), {"a", "b"}):
        print("NOTE: both sides use the same interpreter and the same architecture "
              "source; this measures run-to-run determinism only", flush=True)

    d = a.keep_dir or tempfile.mkdtemp(prefix="runtime_equiv_")
    os.makedirs(d, exist_ok=True)
    infos, arrs = {}, {}
    for side, py in (("a", a.python_a), ("b", a.python_b)):
        out = os.path.join(d, f"logits_{side}.npy")
        print(f"[runtime-equiv] side {side}: {py}", flush=True)
        info, err = _run_side(py, a.model, out, a.tokens, a.chunk, side in a.knurlogic)
        if info is None:
            print(f"FAIL: side {side} did not run: {err}", file=sys.stderr)
            return 2
        infos[side], arrs[side] = info, np.load(out)
    res = compare(arrs["a"], arrs["b"])
    print(report(infos["a"], infos["b"], res))
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(
            {"model": a.model, "tokens": a.tokens, "chunk": a.chunk,
             "a": {"python": a.python_a, **infos["a"]},
             "b": {"python": a.python_b, **infos["b"]}, "result": res}, indent=1))
        print(f"-> {a.out}")
    return 0 if res["bitwise"] else 1


if __name__ == "__main__":
    sys.exit(main())
