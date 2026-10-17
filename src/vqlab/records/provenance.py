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
    vqlab provenance <artifact> --portable [--dry-run]
                                            # strip local paths from old records
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
from vqlab import _layout, config
from vqlab.core.artifacts.hub import is_backup  # local backups (.bak, .pre-*) are not outputs
from vqlab.util.fs import GIB, read_jsonl, write_text_atomic
from vqlab.util.geo import geo_key
from vqlab.util.hashing import HEAD, RECORD, fingerprint, sha256_file

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
# Full-hash files up to this size; above it, head hash + size (the scheme
# artifact_manifest.py already uses). Overridable because a 400 GiB teacher
# is not worth the wall time, but a 5 GiB rewritten shard is.
FULL_MAX = int(os.environ.get("VQLAB_PROV_FULL_HASH_MAX", str(8 << 30)))
# The checkout this package runs from (the folder holding src/vqlab); a
# wheel install has none, and git is then never asked.
REPO = _layout.SRC.parent


# ---------------------------------------------------------------- hashing
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
        rec["sha256"] = sha256_file(rp)
    else:
        rec["head_sha256"] = sha256_file(rp, HEAD)
    return rec


# ------------------------------------------------------------- environment
def _in_checkout() -> bool:
    return (REPO / ".git").exists() and (REPO / "src" / "vqlab").is_dir()


def _git(*a):
    """git's stdout in the checkout, or "" (a wheel install: no checkout, no
    subprocess)."""
    if not _in_checkout():
        return ""
    try:
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True,
                              text=True, timeout=10).stdout.strip("\n")
    except Exception:
        return ""


def _version():
    try:
        import vqlab
        return vqlab.__version__
    except Exception:
        return None


def code_state(script=None) -> dict:
    dirty = [l[3:] for l in _git("status", "--porcelain", "--", "src").splitlines()]
    # Code IDENTITY only (commit, dirty flag, package version): never the
    # checkout's path, which names one machine's disk and ships with the
    # artifact.
    out = {"commit": _git("rev-parse", "HEAD") or None,
           "dirty": bool(dirty), "dirty_files": dirty,
           "version": _version()}
    if script:
        out["script"] = os.path.basename(script)
        out["script_sha256"] = sha256_file(script)
    return out


def env_state() -> dict:
    # no hostname: the record ships with the artifact, and a machine name is
    # as private as its paths (records before 2026-10-10 carry `host`;
    # `--portable` removes it)
    out = {"python": platform.python_version(), "platform": platform.platform()}
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
        {f.name: [os.path.getsize(os.path.realpath(f)), sha256_file(os.path.realpath(f), HEAD)]
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
        from vqlab.runtime import runtime_profile as rpf
        profile, flags = rpf.profile_of(txt), rpf.flags_of(txt)
    except Exception:
        profile, flags = None, {}
    return {"model_py_sha256": sha256_file(mp), "model_py_md5": sha256_file(mp, algo="md5"),
            "profile": profile or "custom", "flags": flags}


# ----------------------------------------------------------------- writing
def _canon(rec):
    return json.dumps({k: v for k, v in rec.items() if k not in ("id", "portable_id")},
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
            h.write(json.dumps(portable_record(old), sort_keys=True) + "\n")
    outs = {}
    for f in sorted(out.iterdir()):
        if f.name in NOT_OUTPUTS + JUNK or is_backup(f.name) or f.name.startswith("._") or f.is_dir():
            continue
        full = (f.name in full_hash) if full_hash is not None else True
        outs[f.name] = file_record(f, full=full)
    rec["outputs"] = outs
    # every recorded path in its portable form (config.portable_text): the
    # record ships with the artifact, so it must not carry this box's disk
    # layout, home directory or checkout location.
    rec = config.portable_obj(rec)
    rec["id"] = hashlib.sha256(_canon(rec).encode()).hexdigest()
    (out / RECORD).write_text(json.dumps(rec, indent=1, default=str))
    print(f"PROVENANCE {rec['id'][:12]} -> {out / RECORD}", flush=True)
    return rec


# ----------------------------------------------------------------- reading
def id_ok(rec) -> bool:
    """The record's content still hashes to its `id` -- or, for a record
    whose paths were made portable after it was written (`--portable`),
    to its `portable_id`. `id` is never rewritten: children and history
    name their parents by it."""
    h = hashlib.sha256(_canon(rec).encode()).hexdigest()
    return h == rec.get("id") or (rec.get("portable_id") is not None
                                  and h == rec["portable_id"])


def portable_record(rec: dict, prefixes=None) -> dict:
    """`rec` with every path string in portable form. Hash-bearing values
    (id, previous, fingerprints, sha256s) are hex, never paths, so they pass
    through untouched. If anything changed, `id` is KEPT (it is what other
    records link by) and `portable_id` names the rewritten content."""
    # an older record names its checkout in code.repo (the lab's checkout,
    # or a queue's pinned worktree): that path and everything under it is
    # the CODE, which the commit already identifies.
    repo = (rec.get("code") or {}).get("repo")
    if isinstance(repo, str) and repo.startswith("/"):
        prefixes = sorted([(repo.rstrip("/"), config.CHECKOUT_LABEL),
                           *(prefixes if prefixes is not None else config.private_prefixes())],
                          key=lambda x: -len(x[0]))
    new = config.portable_obj(rec, prefixes)
    if isinstance(new.get("env"), dict) and "host" in new["env"]:
        new["env"] = {k: v for k, v in new["env"].items() if k != "host"}
    if new == rec:
        return rec
    new["id"] = rec.get("id")
    if "id" in rec:
        new["portable_id"] = hashlib.sha256(_canon(new).encode()).hexdigest()
    return new


def _portable_config_text(text: str, prefixes) -> str:
    """config.json with its `vq_skipzero` source fields (top level and each
    `incremental[*]`, written by skipzero/sz_pack.py) made portable. Edited
    as TEXT, one string literal at a time, so every other byte -- key order,
    indentation, number spelling -- stays as written."""
    cfg = json.loads(text)
    sz = cfg.get("vq_skipzero") if isinstance(cfg, dict) else None
    if not isinstance(sz, dict):
        return text
    vals = [sz.get("source")] + [i.get("source") for i in sz.get("incremental") or []
                                 if isinstance(i, dict)]
    for v in {v for v in vals if isinstance(v, str)}:
        new = config.portable_text(v, prefixes)
        if new != v:
            lit, nlit = json.dumps(v), json.dumps(new)
            text = re.sub(r'("source"\s*:\s*)' + re.escape(lit),
                          lambda m: m.group(1) + nlit, text)
    return text


SANITIZE_NOTE = "sanitized: config.json vq_skipzero.source made portable (paths only)"


def _record_config_sanitize(art, changes, cfg_after: str) -> None:
    """config.json's bytes change, so its recorded hash must too -- as an
    AMENDMENT, never a silent edit: the (portable) record that described the
    old bytes goes to history, and a new record, identical but for that one
    output entry, says what was done and links to it by id. No other hash
    changes. Edits the pending (file, before, after) entries in place."""
    by = {p.name: i for i, (p, _, _) in enumerate(changes)}
    rp, hp = art / RECORD, art / HISTORY
    if not rp.exists():
        return
    cur = json.loads(changes[by[RECORD]][2] if RECORD in by else rp.read_text())
    outs = cur.get("outputs") or {}
    if "config.json" not in outs:
        return
    data = cfg_after.encode()
    new = {k: v for k, v in cur.items() if k != "portable_id"}
    ent = dict(outs["config.json"])
    ent["bytes"] = len(data)
    if "sha256" in ent:
        ent["sha256"] = hashlib.sha256(data).hexdigest()
    if "head_sha256" in ent:
        ent["head_sha256"] = hashlib.sha256(data[:HEAD]).hexdigest()
    new["outputs"] = {**outs, "config.json": ent}
    new["previous"] = cur["id"]
    new["created"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    new["tool"] = {"name": "provenance --portable", "note": SANITIZE_NOTE,
                   "argv": ["vqlab", "provenance", "--portable"],
                   "run_id": os.environ.get("VQLAB_RUN_ID")}
    new["id"] = hashlib.sha256(_canon(new).encode()).hexdigest()
    rec_before = changes[by[RECORD]][1] if RECORD in by else rp.read_text()
    rec_entry = (rp, rec_before, json.dumps(new, indent=1, default=str))
    if RECORD in by:
        changes[by[RECORD]] = rec_entry
    else:
        changes.append(rec_entry)
    line = json.dumps(cur, sort_keys=True) + "\n"
    if HISTORY in by:
        hp_, hb, ha = changes[by[HISTORY]]
        changes[by[HISTORY]] = (hp_, hb, ha + ("" if not ha or ha.endswith("\n") else "\n") + line)
    else:
        hb = hp.read_text() if hp.exists() else ""
        changes.append((hp, hb, hb + ("" if not hb or hb.endswith("\n") else "\n") + line))


def make_portable(art, dry_run=False, prefixes=None) -> list:
    """Rewrite <art>'s record, history and config.json `vq_skipzero`
    source fields so no absolute local path remains
    (paths only; every other byte as written). Returns [(file, before_text,
    after_text)] for each file that changes. Writes `<file>.bak` first.
    Refuses (ValueError) a file whose JSON does not re-serialise to its own
    bytes, since then 'paths only' could not be promised."""
    art = pathlib.Path(art)
    prefixes = prefixes if prefixes is not None else config.private_prefixes()
    # every checkout any record in this artifact names (code.repo) is a
    # checkout in all of them: a later record's cwd may sit in an earlier
    # record's queue worktree
    repos = set()
    for f in (art / RECORD, art / HISTORY):
        if f.exists():
            for ln in f.read_text().splitlines() if f.name == HISTORY else [f.read_text()]:
                if ln.strip():
                    r = (json.loads(ln).get("code") or {}).get("repo")
                    if isinstance(r, str) and r.startswith("/"):
                        repos.add(r.rstrip("/"))
    if repos:
        prefixes = sorted([*((r, config.CHECKOUT_LABEL) for r in repos), *prefixes],
                          key=lambda x: -len(x[0]))
    changes = []
    rp = art / RECORD
    if rp.exists():
        before = rp.read_text()
        rec = json.loads(before)
        if json.dumps(rec, indent=1, default=str) != before:
            raise ValueError(f"{rp}: not in the writer's format; refusing to rewrite")
        new = portable_record(rec, prefixes)
        after = json.dumps(new, indent=1, default=str)
        if after != before:
            changes.append((rp, before, after))
    hp = art / HISTORY
    if hp.exists():
        before = hp.read_text()
        lines = []
        for ln in before.splitlines(keepends=True):
            body = ln.rstrip("\n")
            if not body.strip():
                lines.append(ln)
                continue
            r = json.loads(body)
            if json.dumps(r, sort_keys=True) != body:
                raise ValueError(f"{hp}: a line is not in the writer's format; refusing to rewrite")
            lines.append(json.dumps(portable_record(r, prefixes), sort_keys=True)
                         + ln[len(body):])
        after = "".join(lines)
        if after != before:
            changes.append((hp, before, after))
    cp = art / "config.json"
    if cp.exists():
        before = cp.read_text()
        after = _portable_config_text(before, prefixes)
        if after != before:
            changes.append((cp, before, after))
            _record_config_sanitize(art, changes, after)
    if not dry_run:
        for p, before, after in changes:
            bak = p.with_name(p.name + ".bak")
            bak.write_text(before)
            write_text_atomic(p, after)
    return changes


def load(art) -> dict:
    p = pathlib.Path(art) / RECORD
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())


def verify(art, rec) -> list:
    """[(file, problem)] -- empty means the outputs still match the record."""
    art = pathlib.Path(art)
    bad = []
    if not id_ok(rec):
        bad.append((RECORD, "record edited after it was written (id mismatch)"))
    now = {f.name for f in art.iterdir() if f.is_file() and f.name not in NOT_OUTPUTS + JUNK and not is_backup(f.name)
           and not f.name.startswith(".")}
    for name in sorted(now - set(rec["outputs"])):
        bad.append((name, "added after build"))
    for name, want in rec["outputs"].items():
        if name in JUNK or name.startswith(".") or is_backup(name):
            continue                       # recorded by an older build; not an output
        f = art / name
        if not f.exists():
            bad.append((name, "missing"))
            continue
        rp = pathlib.Path(os.path.realpath(f))
        if rp.stat().st_size != want["bytes"]:
            bad.append((name, f"bytes {rp.stat().st_size} != {want['bytes']}"))
        elif "sha256" in want and sha256_file(rp) != want["sha256"]:
            bad.append((name, "sha256 differs"))
        elif "head_sha256" in want and sha256_file(rp, HEAD) != want["head_sha256"]:
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
            key = (v.get("origin"), geo_key(v.get('dim'), v.get('k')))
            by[key] = by.get(key, 0) + 1
        L.append("  modules   " + ", ".join(f"{n} {o} {g}" for (o, g), n in sorted(by.items())))
    r = rec.get("runtime")
    if rec.get("previous"):
        L.append(f"  amends    {rec['previous'][:16]}  (full chain: --lineage)")
    L.append("  runtime   " + (f"{r['profile']}  md5 {r['model_py_md5'][:8]}"
                               if r else "no model.py bundled"))
    o = rec["outputs"]
    L.append(f"  outputs   {len(o)} files, {sum(v['bytes'] for v in o.values())/GIB:.2f} GiB, "
             f"{sum('symlink_to' in v for v in o.values())} carried over by symlink")
    return "\n".join(L)


def records(art) -> tuple:
    """(records oldest first, recipe record or None) from the build record
    and its history file."""
    recs = []
    art = pathlib.Path(art)
    h = art / HISTORY
    if h.exists():
        recs += [json.loads(x) for x in h.read_text().splitlines() if x.strip()]
    if (art / RECORD).exists():
        recs.append(load(art))
    recipe = next((r for r in recs if r["tool"]["name"].split()[0] in
                   ("fit-moe", "geo-build", "mix", "build-dense-vq", "fit-dense")), None)
    return recs, recipe


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
    # strict: a torn history line is a damaged record, said, not skipped
    hist = {h["id"]: h for h in read_jsonl(pathlib.Path(art) / HISTORY, strict=True)}
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
        # records written since 2026-10-10 hold <models>/X; older ones an
        # absolute path. Both resolve.
        lineage(config.resolve_recorded(i["realpath"]), depth + 1, seen)


def amend(art, names, note) -> dict:
    """Re-record the CURRENT bytes of `names` after an edit made outside a
    vqlab tool (a tokenizer swap, a config restored from the Hub), as an
    amendment: the old record goes to history, the new one says what changed
    and why (`note`, required) and links back by id. Other hashes untouched;
    backup names recorded by older builds are dropped. Returns the new record."""
    art = pathlib.Path(art)
    if not note:
        raise ValueError("an amendment needs a --note saying what changed and why")
    cur = load(art)
    outs = {k: v for k, v in cur["outputs"].items() if not is_backup(k)}
    for n in names:
        f = art / n
        if not f.is_file():
            raise FileNotFoundError(f"{n}: not in {art}")
        outs[n] = file_record(f, full="sha256" in outs.get(n, {"sha256": 1}))
    new = {k: v for k, v in cur.items() if k != "portable_id"}
    new["outputs"] = outs
    new["previous"] = cur["id"]
    new["created"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    new["tool"] = {"name": "provenance --amend", "note": note,
                   "argv": ["vqlab", "provenance", "--amend", *names],
                   "run_id": os.environ.get("VQLAB_RUN_ID")}
    new["id"] = hashlib.sha256(_canon(new).encode()).hexdigest()
    hp = art / HISTORY
    hb = hp.read_text() if hp.exists() else ""
    hp.write_text(hb + ("" if not hb or hb.endswith("\n") else "\n") + json.dumps(cur, sort_keys=True) + "\n")
    (art / RECORD).write_text(json.dumps(new, indent=1, default=str))
    return new


def _portable_main(art, dry_run) -> int:
    import difflib
    try:
        changes = make_portable(art, dry_run=dry_run)
    except ValueError as e:
        print(f"REFUSED: {e}")
        return 1
    if not changes:
        print(f"ok {art}: no absolute local paths in its build records")
        return 0
    for p, before, after in changes:
        diff = [l for l in difflib.unified_diff(before.splitlines(), after.splitlines(),
                                               str(p), str(p) + " (portable)", n=0, lineterm="")]
        nchg = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
        print(f"== {p.name}: {nchg} line(s) change")
        for l in diff:
            print(l if len(l) < 240 else l[:237] + "...")
    print(f"{'DRY RUN, nothing written' if dry_run else 'REWRITTEN (.bak kept)'}: "
          f"{len(changes)} file(s) in {art}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vqlab provenance",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("artifact")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--json", action="store_true")
    g.add_argument("--verify", action="store_true")
    g.add_argument("--lineage", action="store_true")
    g.add_argument("--portable", action="store_true",
                   help="rewrite the record + history so no absolute local path "
                        "remains (paths only; .bak written first)")
    g.add_argument("--amend", nargs="+", metavar="FILE",
                   help="re-record these files' current bytes as an amendment "
                        "(requires --note)")
    ap.add_argument("--note", default="", help="with --amend: what changed and why")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --portable: print the diff, write nothing")
    a = ap.parse_args(argv)
    if a.portable:
        return _portable_main(a.artifact, a.dry_run)
    if a.amend:
        try:
            new = amend(a.artifact, a.amend, a.note)
        except (ValueError, FileNotFoundError) as e:
            print(f"REFUSED: {e}")
            return 1
        print(f"amended {a.artifact}: {new['id'][:12]} (was {new['previous'][:12]})")
        bad = verify(a.artifact, new)
        for n, why in bad:
            print(f"  DRIFT  {n}: {why}")
        return 2 if bad else 0
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
