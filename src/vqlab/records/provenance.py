#!/usr/bin/env python3
"""vqlab provenance — the BUILD RECORD an artifact carries: how it was made.

`vqlab manifest` (artifact_manifest.py) answers "were these bytes rewritten?"
-- a tamper stamp: shard sizes, mtimes, head hashes, kept OUTSIDE the
artifact. It says nothing about how the bytes were MADE. Reconstructing that
for the paper v5 revision (2026-09-23..25) took a day of reading EXPERIMENTS.md
E92-E118 by hand: three fitter vintages inside one 397B family, geo-build's
default scale<->codebook alternation (F80/F81) on nine repaired modules per
rung, three different runtime bundles across the same family, and a rebuilt
35B struct base that was not byte-identical to the published one under the
"same recipe" (an mlx version change moved scales up to 12.8%).
docs/PROVENANCE.md is the design; this is its first slice.

Every build tool calls `write_build_record(...)` as its LAST step. It writes
`vqlab_provenance.json` INTO the new artifact directory (a build output is a
fresh directory, so the record cannot describe bytes it later alters) with:

  tool      name, exact argv, cwd, which args were left AT DEFAULT (a default
            that flips -- fit-moe init 08-18, geo-build alternation -- is then
            visible in the record, not only in git history)
  code      repo commit, dirty flag + dirty files, sha256 of the tool script
  env       python / mlx / mlx-lm / numpy versions, host, platform
  method    the tool's fitter settings, written by the tool itself
  inputs    each parent (teacher, base artifact, reuse dirs) with its
            fingerprint and, when it has one, ITS provenance id -> lineage
  modules   per-module geometry and ORIGIN (fit / reused-from / resumed)
  runtime   bundled model.py sha256 + md5 + profile (v1.5 / v2) + VQ_* flags
  outputs   every file: bytes, sha256 (or head hash above a size cap),
            symlink target for shards carried over unchanged

The record's `id` is the sha256 of its own canonical JSON (minus `id`), so a
child names its parent by content, not by a path that another session can
rebundle underneath it.

    vqlab provenance <artifact>             # human summary
    vqlab provenance <artifact> --json      # the record
    vqlab provenance <artifact> --verify    # re-hash outputs against it (rc 2 on drift)
    vqlab provenance <artifact> --lineage   # walk parents that carry records
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys

RECORD = "vqlab_provenance.json"
# Records this one REPLACED, oldest first: an in-place tool (bundle,
# rebundle, graft, patch) amends the artifact, and the chain of what it was
# before travels with it instead of being overwritten.
HISTORY = "vqlab_provenance.history.jsonl"
NOT_OUTPUTS = (RECORD, HISTORY)
# Local files nobody built: Finder rewrites .DS_Store whenever the folder is
# opened, which would otherwise fail every record it lands in. check-release
# refuses to ship them.
JUNK = (".DS_Store",)
SCHEMA = "vqlab.provenance/1"
HEAD = 1 << 20
# Full-hash files up to this size; above it, head hash + size (the scheme
# artifact_manifest.py already uses). Overridable because a 400 GiB teacher
# is not worth the wall time, but a 5 GiB rewritten shard is.
FULL_MAX = int(os.environ.get("VQLAB_PROV_FULL_HASH_MAX", str(8 << 30)))
REPO = pathlib.Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------- hashing
def _sha(path, limit=None, algo="sha256"):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        if limit is not None:
            h.update(f.read(limit))
        else:
            for blk in iter(lambda: f.read(1 << 24), b""):
                h.update(blk)
    return h.hexdigest()


def file_record(p: pathlib.Path, full=True):
    """bytes + hash; symlinks record their target and are head-hashed (they
    are carried-over bytes whose own record is the parent's)."""
    rp = pathlib.Path(os.path.realpath(p))
    n = rp.stat().st_size
    rec = {"bytes": n}
    if p.is_symlink():
        rec["symlink_to"] = str(rp)
        full = False
    if full and n <= FULL_MAX:
        rec["sha256"] = _sha(rp)
    else:
        rec["head_sha256"] = _sha(rp, HEAD)
    return rec


def fingerprint(d) -> dict:
    """Identity of an INPUT directory, cheap enough for a 751 GB teacher:
    per-shard bytes + head hash, full sha of config/index/model.py, its own
    provenance id if it carries one, and the HF revision if it is a hub
    snapshot."""
    d = pathlib.Path(d)
    rp = pathlib.Path(os.path.realpath(d))
    out = {"path": str(d), "realpath": str(rp)}
    parts = rp.parts
    if "snapshots" in parts:                        # HF hub cache layout
        i = parts.index("snapshots")
        if i + 1 < len(parts):
            out["hf_revision"] = parts[i + 1]
            out["hf_repo"] = parts[i - 1].removeprefix("models--").replace("--", "/")
    if not rp.is_dir():
        return out
    rec = rp / RECORD
    if rec.exists():
        out["provenance_id"] = json.loads(rec.read_text()).get("id")
    for name in ("config.json", "model.safetensors.index.json", "model.py"):
        if (rp / name).exists():
            out.setdefault("files", {})[name] = _sha(rp / name)
    shards = {}
    for f in sorted(rp.glob("*.safetensors")):
        r = pathlib.Path(os.path.realpath(f))
        shards[f.name] = {"bytes": r.stat().st_size, "head_sha256": _sha(r, HEAD)}
    out["shards"] = shards
    return out


# ------------------------------------------------------------- environment
def _git(*a):
    try:
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True,
                              text=True, timeout=10).stdout.strip("\n")
    except Exception:
        return ""


def code_state(script=None) -> dict:
    dirty = [l[3:] for l in _git("status", "--porcelain", "--", "src").splitlines()]
    out = {"repo": str(REPO), "commit": _git("rev-parse", "HEAD") or None,
           "dirty": bool(dirty), "dirty_files": dirty}
    if script:
        out["script"] = os.path.basename(script)
        out["script_sha256"] = _sha(script)
    return out


def env_state() -> dict:
    out = {"python": platform.python_version(), "host": platform.node(),
           "platform": platform.platform()}
    for mod in ("mlx", "mlx_lm", "numpy"):
        try:
            if mod == "mlx":
                import mlx.core as m
            else:
                m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:
            out[mod] = None
    return out


def args_state(ap: argparse.ArgumentParser, args: argparse.Namespace) -> dict:
    """Every resolved arg, and which were left at DEFAULT. The second list is
    the point: when a default flips, old records say they relied on it."""
    vals = {k: v for k, v in vars(args).items()}
    at_default = sorted(k for k, v in vals.items() if ap.get_default(k) == v)
    return {"resolved": vals, "at_default": at_default}


def shard_fingerprint(d) -> str:
    """16-hex identity of an artifact's weight bytes: per shard (resolved
    through symlinks, so a pin and its source agree) size + first-MiB
    sha256. The registry and every scorer use this ONE definition."""
    d = pathlib.Path(d)
    shards = sorted(d.glob("*.safetensors"))
    return hashlib.sha256(json.dumps(
        {f.name: [os.path.getsize(os.path.realpath(f)), _sha(os.path.realpath(f), HEAD)]
         for f in shards}, sort_keys=True).encode()).hexdigest()[:16]


def measured(d) -> dict:
    """What a SCORE is a score OF. Every scorer stamps this into its output,
    so a comparison row names its artifact mechanically (AGENTS.md rule III:
    "a comparison row must name the ARTIFACT and the INSTRUMENT").

    build_record  the artifact's own record id (None for pre-record artifacts)
    pin           a pinned copy's state + source, and the SOURCE's record id
    runtime       model.py md5 + profile, and its mtime: re-check it after a
                  long campaign ("an artifact can change under you", F151)
    fingerprint   shard_fingerprint(): same bytes <=> same value
    run_id        the vqlab run that produced the score (the run log line)"""
    d = pathlib.Path(d)
    out = {"path": str(d), "realpath": os.path.realpath(d),
           "run_id": os.environ.get("VQLAB_RUN_ID")}
    if not d.is_dir():
        return out
    rec = d / RECORD
    out["build_record"] = json.loads(rec.read_text()).get("id") if rec.exists() else None
    pm = d / "vqlab_pin.json"
    if pm.exists():
        pin = json.loads(pm.read_text())
        src = pathlib.Path(pin.get("source", ""))
        out["pin"] = {"state": pin.get("state"), "source": str(src), "runtime": pin.get("runtime"),
                      "source_build_record": (json.loads((src / RECORD).read_text()).get("id")
                                              if (src / RECORD).exists() else None)}
    rt = runtime_state(d)
    if rt:
        out["runtime"] = {"model_py_md5": rt["model_py_md5"], "profile": rt["profile"],
                          "model_py_mtime": os.stat(d / "model.py").st_mtime}
    try:
        out["fingerprint"] = shard_fingerprint(d)
    except OSError as e:
        out["fingerprint"] = f"unreadable: {e}"
    return out


def runtime_state(art: pathlib.Path):
    mp = art / "model.py"
    if not mp.exists():
        return None
    txt = mp.read_text()
    try:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # src/
        from vqlab import _layout  # noqa: E402,F401  one module object per name
        import runtime_profile as rpf
        profile, flags = rpf.profile_of(txt), rpf.flags_of(txt)
    except Exception:
        profile, flags = None, {}
    return {"model_py_sha256": _sha(mp), "model_py_md5": _sha(mp, algo="md5"),
            "profile": profile or "custom", "flags": flags}


# ----------------------------------------------------------------- writing
def _canon(rec):
    return json.dumps({k: v for k, v in rec.items() if k != "id"},
                      sort_keys=True, separators=(",", ":"), default=str)


def write_build_record(out, *, tool, script=None, ap=None, args=None,
                       method=None, inputs=(), modules=None, full_hash=None,
                       argv=None):
    """Write <out>/vqlab_provenance.json and return the record.

    inputs:  [(role, path), ...]  role in teacher|base|reuse|source|...
    modules: {module: {"dim":.., "k":.., "origin": "fit"|"reuse"|"resume", ...}}
    full_hash: set of output filenames to full-hash (default: every regular,
               non-symlinked file -- i.e. what THIS run wrote).
    """
    out = pathlib.Path(out)
    rec = {
        "schema": SCHEMA,
        "created": datetime.datetime.now(datetime.timezone.utc)
                   .isoformat(timespec="seconds"),
        "tool": {"name": tool, "argv": list(argv if argv is not None else sys.argv),
                 "cwd": os.getcwd(),
                 "run_id": os.environ.get("VQLAB_RUN_ID")},
        "code": code_state(script),
        "env": env_state(),
        "method": method or {},
        "inputs": [{"role": r, **fingerprint(p)} for r, p in inputs],
    }
    if ap is not None and args is not None:
        rec["tool"]["args"] = args_state(ap, args)
    if modules is not None:
        rec["modules"] = modules
    rec["runtime"] = runtime_state(out)
    prior = out / RECORD
    if prior.exists():
        # AMENDMENT: keep what this artifact was, and link to it by id
        old = json.loads(prior.read_text())
        rec["previous"] = old.get("id")
        with open(out / HISTORY, "a") as h:
            h.write(json.dumps(old, sort_keys=True) + "\n")
    outs = {}
    for f in sorted(out.iterdir()):
        if f.name in NOT_OUTPUTS + JUNK or f.name.startswith("._") or f.is_dir():
            continue
        full = (f.name in full_hash) if full_hash is not None else True
        outs[f.name] = file_record(f, full=full)
    rec["outputs"] = outs
    rec["id"] = hashlib.sha256(_canon(rec).encode()).hexdigest()
    (out / RECORD).write_text(json.dumps(rec, indent=1, default=str))
    print(f"PROVENANCE {rec['id'][:12]} -> {out / RECORD}", flush=True)
    return rec


# ----------------------------------------------------------------- reading
def load(art) -> dict:
    p = pathlib.Path(art) / RECORD
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())


def verify(art, rec) -> list:
    """[(file, problem)] -- empty means the outputs still match the record."""
    art = pathlib.Path(art)
    bad = []
    if hashlib.sha256(_canon(rec).encode()).hexdigest() != rec.get("id"):
        bad.append((RECORD, "record edited after it was written (id mismatch)"))
    now = {f.name for f in art.iterdir() if f.is_file() and f.name not in NOT_OUTPUTS + JUNK
           and not f.name.startswith("._")}
    for name in sorted(now - set(rec["outputs"])):
        bad.append((name, "added after build"))
    for name, want in rec["outputs"].items():
        if name in JUNK or name.startswith("._"):
            continue                       # recorded by an older build; not an output
        f = art / name
        if not f.exists():
            bad.append((name, "missing"))
            continue
        rp = pathlib.Path(os.path.realpath(f))
        if rp.stat().st_size != want["bytes"]:
            bad.append((name, f"bytes {rp.stat().st_size} != {want['bytes']}"))
        elif "sha256" in want and _sha(rp) != want["sha256"]:
            bad.append((name, "sha256 differs"))
        elif "head_sha256" in want and _sha(rp, HEAD) != want["head_sha256"]:
            bad.append((name, "head sha256 differs"))
    return bad


def summary(art, rec) -> str:
    t, c, e = rec["tool"], rec["code"], rec["env"]
    L = [f"{art}",
         f"  id        {rec['id'][:16]}   ({rec['schema']}, {rec['created']})",
         f"  tool      {t['name']}   commit {str(c.get('commit'))[:10]}"
         f"{'  DIRTY: ' + ', '.join(c['dirty_files'][:4]) if c.get('dirty') else ''}",
         f"  env       mlx {e.get('mlx')}  mlx-lm {e.get('mlx_lm')}  "
         f"py {e.get('python')}  host {e.get('host')}"]
    m = rec.get("method") or {}
    if m:
        L.append("  method    " + ", ".join(f"{k}={v}" for k, v in m.items()))
    dflt = (t.get("args") or {}).get("at_default") or []
    if dflt:
        L.append("  defaults  " + ", ".join(dflt))
    for i in rec.get("inputs", []):
        tag = (f"prov {i['provenance_id'][:12]}" if i.get("provenance_id")
               else f"hf {i['hf_repo']}@{i['hf_revision'][:10]}" if i.get("hf_revision")
               else f"{len(i.get('shards', {}))} shards, NO RECORD")
        L.append(f"  input     {i['role']:8s} {i['path']}  [{tag}]")
    mods = rec.get("modules") or {}
    if mods:
        by = {}
        for v in mods.values():
            key = (v.get("origin"), f"d{v.get('dim')}-K{v.get('k')}")
            by[key] = by.get(key, 0) + 1
        L.append("  modules   " + ", ".join(f"{n} {o} {g}" for (o, g), n in sorted(by.items())))
    r = rec.get("runtime")
    if rec.get("previous"):
        L.append(f"  amends    {rec['previous'][:16]}  (full chain: --lineage)")
    L.append("  runtime   " + (f"{r['profile']}  md5 {r['model_py_md5'][:8]}"
                               if r else "no model.py bundled"))
    o = rec["outputs"]
    L.append(f"  outputs   {len(o)} files, {sum(v['bytes'] for v in o.values())/2**30:.2f} GiB, "
             f"{sum('symlink_to' in v for v in o.values())} carried over by symlink")
    return "\n".join(L)


def lineage(art, depth=0, seen=None):
    seen = seen if seen is not None else set()
    try:
        rec = load(art)
    except FileNotFoundError:
        print("  " * depth + f"{art}  [no provenance record: lineage stops here]")
        return
    print("  " * depth + f"{art}  [{rec['tool']['name']} {rec['id'][:12]}]")
    if rec["id"] in seen:
        return
    seen.add(rec["id"])
    hist = {}
    hp = pathlib.Path(art) / HISTORY
    if hp.exists():
        for line in hp.read_text().splitlines():
            h = json.loads(line)
            hist[h["id"]] = h
    prev, d = rec.get("previous"), depth + 1
    chain = [rec]
    while prev:                            # in-place amendments, newest first
        h = hist.get(prev)
        if not h:
            print("  " * d + f"amends {prev[:12]}  [record not in history]")
            break
        print("  " * d + f"amends {h['tool']['name']} {h['id'][:12]} ({h['created']})")
        chain.append(h)
        prev = h.get("previous")
    # every record in the chain contributes its inputs (an amendment made in
    # place usually has none of its own; the build it amends does)
    ins = [i for r in chain for i in r.get("inputs", [])]
    for i in ins:
        if i["role"] == "teacher":
            print("  " * (depth + 1) + f"teacher {i['path']}")
            continue
        lineage(i["realpath"], depth + 1, seen)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vqlab provenance",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("artifact")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--json", action="store_true")
    g.add_argument("--verify", action="store_true")
    g.add_argument("--lineage", action="store_true")
    a = ap.parse_args(argv)
    try:
        rec = load(a.artifact)
    except FileNotFoundError:
        print(f"NO RECORD: {a.artifact} has no {RECORD}. Built before provenance "
              f"records existed, or by a tool that does not write one yet "
              f"(docs/PROVENANCE.md lists which). `vqlab manifest` may still "
              f"have a tamper stamp for it.")
        return 1
    if a.json:
        print(json.dumps(rec, indent=1))
    elif a.lineage:
        lineage(a.artifact)
    elif a.verify:
        bad = verify(a.artifact, rec)
        for n, why in bad:
            print(f"  DRIFT  {n}: {why}")
        print(f"{'CHANGED' if bad else 'ok'} {a.artifact}: "
              f"{len(rec['outputs'])} recorded files, {len(bad)} problems")
        return 2 if bad else 0
    else:
        print(summary(a.artifact, rec))
    return 0


if __name__ == "__main__":
    sys.exit(main())
