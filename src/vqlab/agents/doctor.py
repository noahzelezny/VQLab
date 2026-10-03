#!/usr/bin/env python3
"""vqlab doctor: which interpreter, packages, storage and credentials this run would use.

    vqlab doctor            # report; exit 1 if anything would stop a fit or a score
    vqlab doctor --json

Operators lost hours on 2026-10-02 to commands that ran under the wrong
interpreter, without `vqlab` on PATH, or without the Hugging Face token.
This prints every one of those facts, from the running process, and touches
no GPU (it reads the Metal device description, never allocates on it).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import sys


# architectures the lab's families load through mlx-lm (the family's model_type)
ARCHES = ("deepseek_v4", "qwen3_5_moe", "qwen3_5", "gemma4", "glm4_moe", "qwen4_exp")


def _pkg(name):
    try:
        from importlib.metadata import version
        v = version(name)
    except Exception:
        v = None
    spec = importlib.util.find_spec(name.replace("-", "_"))
    path = None
    if spec and spec.origin:
        path = str(pathlib.Path(spec.origin).parent)
    elif spec and spec.submodule_search_locations:          # namespace package (mlx)
        path = str(list(spec.submodule_search_locations)[0])
    return {"version": v, "path": path}


def _sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:12]


def _hf_token():
    if os.environ.get("HF_TOKEN"):
        return "$HF_TOKEN"
    home = pathlib.Path(os.environ.get("HF_HOME", pathlib.Path.home() / ".cache/huggingface"))
    return str(home / "token") if (home / "token").exists() else None


def report():
    import vqlab
    from vqlab import config as C
    r = {"python": sys.executable, "vqlab": {"version": getattr(vqlab, "__version__", None),
                                              "path": str(pathlib.Path(vqlab.__file__).parent)},
         "vqlab_on_path": shutil.which("vqlab"), "packages": {}, "arch_files": {}, "storage": {},
         "hf_home": os.environ.get("HF_HOME"), "hf_token": _hf_token(), "problems": [], "notes": []}
    src = pathlib.Path(vqlab.__file__).parent.parent
    installed = _pkg("vqlab")["version"] is not None
    if (src.parent / "pyproject.toml").exists() and "site-packages" not in str(src):
        r["notes"].append(f"vqlab runs from the checkout {src}"
                          + (" (editable install)" if installed else
                             " via PYTHONPATH, not installed: `pip install -e . --no-deps`"))
    own = pathlib.Path(sys.executable).parent / "vqlab"
    if own.exists() and r["vqlab_on_path"] != str(own):
        r["notes"].append(f"this interpreter's `vqlab` is {own}, but the shell's PATH "
                          f"{'finds ' + r['vqlab_on_path'] if r['vqlab_on_path'] else 'does not reach it'}: "
                          f"activate the environment, or call {own} directly")
    elif not own.exists() and not r["vqlab_on_path"]:
        r["notes"].append("`vqlab` is not installed in this interpreter: `pip install -e . --no-deps`, "
                          "or run `python -m vqlab.cli`")
    for p in ("mlx", "mlx-lm", "mlx-vlm", "knurlogic"):
        r["packages"][p] = _pkg(p)
    if not r["packages"]["mlx"]["version"]:
        r["problems"].append("mlx is not importable in this interpreter: nothing can fit or score")
    try:
        import mlx.core as mx
        dev_info = getattr(mx, "device_info", None) or mx.metal.device_info
        info = dev_info() if mx.metal.is_available() else {}
        r["metal"] = {"available": mx.metal.is_available(), "device": info.get("device_name"),
                      "memory_gib": round(info.get("memory_size", 0) / 2**30)}
    except Exception as e:
        r["metal"] = {"available": False, "error": str(e)}
    mlm = r["packages"]["mlx-lm"]["path"]
    if mlm:
        for arch in ARCHES:
            try:
                spec = importlib.util.find_spec(f"mlx_lm.models.{arch}")
            except Exception:
                spec = None
            r["arch_files"][arch] = _sha(spec.origin) if spec and spec.origin else None
            if r["arch_files"][arch] is None:
                r["notes"].append(f"mlx-lm has no {arch} architecture: VQLab cannot load, score or "
                                  f"convert that family in this interpreter")
    for key, fn in (("scratch", C.scratch), ("models", C.models), ("teachers", C.teachers)):
        p = fn()
        ok = p.exists()
        free = shutil.disk_usage(p).free / 2**30 if ok else None
        r["storage"][key] = {"path": str(p), "exists": ok,
                             "writable": ok and os.access(p, os.W_OK),
                             "free_gib": round(free, 1) if free is not None else None}
        if not ok:
            r["problems"].append(f"{key} root {p} does not exist (unmounted volume?)")
    if not r["hf_token"]:
        r["notes"].append("no Hugging Face token (HF_TOKEN or $HF_HOME/token): downloads of gated "
                          "repos and publish will fail")
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vqlab doctor", description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    r = report()
    if a.json:
        print(json.dumps(r, indent=1))
        return 1 if r["problems"] else 0
    print(f"python      {r['python']}")
    print(f"vqlab       {r['vqlab']['version']}  {r['vqlab']['path']}")
    print(f"on PATH     {r['vqlab_on_path'] or '-'}")
    for p, v in r["packages"].items():
        print(f"{p:11s} {v['version'] or '-':12s} {v['path'] or ''}")
    m = r["metal"]
    print(f"metal       {m.get('device') or '-'}  {m.get('memory_gib', '-')} GiB" if m.get("available")
          else f"metal       unavailable {m.get('error', '')}")
    for k, v in r["arch_files"].items():
        print(f"arch        {k}.py " + (f"sha {v}" if v else "MISSING"))
    for k, v in r["storage"].items():
        print(f"{k:11s} {v['path']}  " + (f"{v['free_gib']} GiB free" if v["exists"] else "MISSING"))
    print(f"HF_HOME     {r['hf_home'] or '-'}  token: {'yes' if r['hf_token'] else 'no'}")
    for n in r["notes"]:
        print(f"note: {n}")
    for p in r["problems"]:
        print(f"PROBLEM: {p}")
    return 1 if r["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
