#!/usr/bin/env python3
"""The NUMERICS BUILD a number was computed under, and the check that two match.

F194: scoring (mlx 0.32.0.dev + a grafted mlx-lm 0.31.9) and serving (mlx
0.31.2) disagree by up to 1.0 logit on DeepSeek while the architecture code is
bitwise identical. A KL number is only meaningful for the build that produced
it, so every teacher cache, every scored record and every per-position array
carries this stamp:

    mlx             mlx's version (the kernels)
    mlx_lm          mlx-lm's version (the runtime that builds the graph)
    arch_file       the runtime library's architecture module for the
                    model_type (mlx_lm.models.<mt> or mlx_vlm.models.<mt>), as
                    RESOLVED in this interpreter -- a vendored module already
                    in sys.modules (Knurlogic's register) wins, as it does at
                    load time
    arch_sha256     its sha256
    scorer_variant  the family plugin's scorer variant (None without one)

The ARCH file, not the artifact's bundled model.py, is what is compared: a
teacher and its VQ student load different model.py files by design (the
bundle splices the VQ runtime around `mlx_lm.models.<mt>`), so their bundles
never match while the architecture they both call must. The file the model
class actually came from is recorded beside it (`loaded_file`/`loaded_sha256`)
and is not compared.

    python -m vqlab.core.numerics --model <dir>    # this interpreter's build, JSON

Reads files and package metadata only; never touches the GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import pathlib
import sys

# The fields a cache and a run must agree on. Order is the report order.
COMPARED = ("mlx", "mlx_lm", "arch_sha256", "scorer_variant")
_UNSET = object()


def _sha(path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 22), b""):
                h.update(blk)
        return h.hexdigest()
    except OSError:
        return None


def _version(dist: str, module: str) -> str | None:
    # The imported module's own __version__ first: a grafted install can keep
    # the pip metadata of the build it replaced.
    mod = sys.modules.get(module)
    v = getattr(mod, "__version__", None) if mod else None
    if v:
        return str(v)
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _knurlogic() -> str | None:
    """knurlogic's version + commit (its architecture files are the ones
    loaded). Recorded, not compared: arch_sha256 already compares the bytes
    that matter, and a knurlogic commit that leaves them alone changes no
    number."""
    try:
        d = importlib.metadata.distribution("knurlogic")
    except importlib.metadata.PackageNotFoundError:
        return None
    commit = None
    try:
        commit = (json.loads(d.read_text("direct_url.json") or "{}")
                  .get("vcs_info", {}).get("commit_id"))
    except ValueError:
        pass
    return d.version + (f"@{commit[:9]}" if commit else "")


def versions() -> dict:
    return {"mlx": _version("mlx", "mlx.core") or _version("mlx", "mlx"),
            "mlx_lm": _version("mlx-lm", "mlx_lm"),
            "knurlogic": _knurlogic()}


def model_type_of(model_dir) -> str | None:
    cfg = json.loads((pathlib.Path(model_dir) / "config.json").read_text())
    return cfg.get("model_type") or cfg.get("text_config", {}).get("model_type")


def _runtime_for(family: str | None) -> str:
    if not family:
        return "mlx_lm"
    try:
        from vqlab import _layout  # noqa: F401  bare sibling names
        import runtime_load
        return runtime_load.runtime_for(family)
    except Exception:                                # noqa: BLE001
        return "mlx_lm"


def arch_file(model_type: str | None, runtime: str = "mlx_lm") -> str | None:
    """Path of the architecture module this interpreter resolves for model_type."""
    if not model_type:
        return None
    mt = model_type
    if runtime == "mlx_lm":
        try:
            from mlx_lm.utils import MODEL_REMAPPING
            mt = MODEL_REMAPPING.get(mt, mt)
        except Exception:                            # noqa: BLE001
            pass
    name = f"{runtime}.models.{mt}"
    mod = sys.modules.get(name)
    if mod is not None:
        return getattr(mod, "__file__", None)
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    return spec.origin if spec else None


def scorer_variant(model_type: str | None):
    if not model_type:
        return None
    try:
        from vqlab.family import variant_for
        return variant_for(model_type)
    except Exception:                                # noqa: BLE001
        return None


def build(model_type: str | None = None, *, family: str | None = None,
          model=None, model_dir=None, variant=_UNSET) -> dict:
    """This interpreter's numerics build for a model_type.

    `variant` defaults to the family plugin's current scorer variant; a path
    that does not apply the variant (kl_damage's direct forward) passes None.
    `model`, when loaded, adds the file its class actually came from."""
    if model_type is None and model_dir is not None:
        model_type = model_type_of(model_dir)
    runtime = _runtime_for(family)
    af = arch_file(model_type, runtime)
    out = {**versions(), "model_type": model_type, "runtime": runtime,
           "arch_file": af, "arch_sha256": _sha(af) if af else None,
           "scorer_variant": scorer_variant(model_type) if variant is _UNSET else variant}
    if model is not None:
        mod = sys.modules.get(type(model).__module__)
        lf = getattr(mod, "__file__", None)
        if lf:
            out["loaded_file"], out["loaded_sha256"] = lf, _sha(lf)
    return out


def diff(a: dict | None, b: dict | None) -> list:
    """[(field, a, b)] for every COMPARED field that differs."""
    a, b = a or {}, b or {}
    return [(k, a.get(k), b.get(k)) for k in COMPARED if a.get(k) != b.get(k)]


def describe(st: dict | None) -> str:
    """'mlx X, mlx-lm Y, variant Z' -- the card line."""
    if not st:
        return "build not stamped"
    return (f"mlx {st.get('mlx')}, mlx-lm {st.get('mlx_lm')}, "
            + (f"knurlogic {st['knurlogic']}, " if st.get("knurlogic") else "")
            + f"variant {st.get('scorer_variant') or 'none'}")


def check_cache(meta: dict, run: dict, allow: bool = False, what: str = "teacher cache") -> dict:
    """Refuse when a cache's build differs from this run's; return the record.

    A cache written before the stamp existed (no `numerics`) is WARNED about,
    not refused: only its top-level scorer_variant (stamped since F195) can be
    checked. `allow` turns a refusal into a recorded override."""
    cached = meta.get("numerics")
    if cached is None:
        msg = (f"WARNING: the {what} carries NO numerics build stamp (written before "
               f"2026-10-03). mlx, mlx-lm and the arch file cannot be checked; this run "
               f"is {describe(run)}. Rebuild the cache to make the KL attributable.")
        print(msg, flush=True)
        print(msg, file=sys.stderr, flush=True)
        bad = [("scorer_variant", meta.get("scorer_variant"), run.get("scorer_variant"))]
        bad = [x for x in bad if x[1] != x[2]]
        status = "unstamped-cache"
    else:
        bad = diff(cached, run)
        status = "match"
    if bad:
        lines = "; ".join(f"{k}: cache {c!r} vs run {r!r}" for k, c, r in bad)
        if not allow:
            raise SystemExit(
                f"FAIL: the {what} was built with different numerics than this run ({lines}). "
                "Teacher and student must be computed by the same build, or the KL measures "
                "the build, not the quantization. Rebuild the cache in this environment, or "
                "pass --allow-build-mismatch (recorded in the result).")
        print(f"WARNING: build mismatch ALLOWED by --allow-build-mismatch: {lines}", flush=True)
        status = "mismatch-allowed"
    return {"status": status, "mismatches": [list(x) for x in bad]}


# ------------------------------------------------------------ cache identity
def cache_identity(cache_dir) -> dict:
    """What makes two per-position arrays pairable: the SAME positions
    (tokens.safetensors) against the SAME teacher (its stored logprobs, head-
    hashed: a full-vocab cache is 12 GB) and the build the cache was made by."""
    cd = pathlib.Path(cache_dir)
    out = {"dir": str(cd), "tokens_sha256": _sha(cd / "tokens.safetensors")}
    for name in ("teacher_full.safetensors", "teacher_topk.safetensors"):
        p = cd / name
        if p.exists():
            h = hashlib.sha256()
            with open(p, "rb") as f:
                h.update(f.read(1 << 20))
            out["teacher_file"] = name
            out["teacher_head_sha256"] = h.hexdigest()
            out["teacher_bytes"] = p.stat().st_size
            break
    try:
        out["numerics"] = json.loads((cd / "meta.json").read_text()).get("numerics")
    except (OSError, ValueError):
        out["numerics"] = None
    return out


def pairing_problems(a: dict | None, b: dict | None) -> list:
    """Why two per-position sidecar records can NOT be paired ([] = pairable)."""
    if not a or not b:
        return ["a per-position array has no .json sidecar"]
    probs = []
    ca, cb = a.get("kl_cache"), b.get("kl_cache")
    if not ca or not cb:
        probs.append("a sidecar predates the cache identity stamp (kl_cache)")
    else:
        for k in ("tokens_sha256", "teacher_head_sha256", "teacher_bytes"):
            if ca.get(k) != cb.get(k):
                probs.append(f"different cache: {k} {ca.get(k)!r} vs {cb.get(k)!r}")
    na, nb = a.get("numerics"), b.get("numerics")
    if not na or not nb:
        probs.append("a sidecar carries no numerics build stamp")
    else:
        probs += [f"different build: {k} {x!r} vs {y!r}" for k, x, y in diff(na, nb)]
    if a.get("kl_positions") != b.get("kl_positions"):
        probs.append(f"different position count {a.get('kl_positions')} vs {b.get('kl_positions')}")
    return probs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", help="model dir (its config.json names the model_type)")
    ap.add_argument("--model-type")
    ap.add_argument("--family")
    a = ap.parse_args(argv)
    mt = a.model_type or (model_type_of(a.model) if a.model else None)
    family = a.family
    if family is None and mt:
        try:
            from vqlab import _layout  # noqa: F401
            import runtime_load
            family = runtime_load.family_for_model_type(mt)
        except Exception:                            # noqa: BLE001
            family = None
    print(json.dumps(build(mt, family=family)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
