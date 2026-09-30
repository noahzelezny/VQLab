#!/usr/bin/env python3
"""vqlab registry — one queryable index of every artifact, and Hub drift.

    vqlab registry scan <root> [<root> ...]   READ-ONLY on artifacts; writes
                                              registry/artifacts.jsonl (in git)
    vqlab registry list [--profile v2] [--family qwen3_5] [--grep 35B]
    vqlab registry hub <artifact> [--repo owner/name]
    vqlab registry hub --all                  every registered artifact that
                                              names a Hub repo

WHY. Three facts about the published fleet were reconstructed by hand on
2026-09-23..25 and are exactly what bytes can prove: which runtime bundle
each artifact ships (35B 3.4 = v1.5 while 3.8/4.6/5.4 = v2; three different
397B bundles), what geometry mix it carries, and whether the local serving
copy is the published one (8 of 20 were not, 2026-09-19). A registry line is
built ONLY from those provable facts -- nothing is transcribed from notes,
so nothing in it can be wrong the way a notebook entry can. How an old
artifact was FIT is not provable from its bytes; its build record says so
when it has one, and `recipe` is null when it does not.

Hub drift compares metadata only (no downloads): LFS files by sha256, small
files by git blob sha1 computed locally. For model.py it also reports the
runtime profile on each side, from the local file.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import os
import pathlib
import re
import socket
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # src/
from vqlab import _layout  # noqa: E402,F401  one module object per name
import provenance  # noqa: E402
import runtime_profile  # noqa: E402

REPO = _layout.SRC.parent
REG = REPO / "registry" / "artifacts.jsonl"
HUB = REPO / "registry" / "hub.jsonl"          # latest Hub comparison per repo
HF_OWNER_SEP = "--"


def is_artifact(d: pathlib.Path) -> bool:
    c = d / "config.json"
    if not c.is_file() or not any(d.glob("*.safetensors")):
        return False
    try:
        cfg = json.loads(c.read_text())
    except ValueError:
        return False
    return bool(cfg.get("vq_modules") or cfg.get("vq_linear") or cfg.get("vq_embed")
                or cfg.get("vq_ple"))


def vq_specs(cfg):
    """{module: {dim, k, group?, pack_bits?, kind}} for EVERY VQ tensor a
    config declares. Four config blocks hold them: vq_modules (MoE experts),
    vq_linear (dense), vq_embed, and vq_ple -- ONE geometry block for all the
    PLE n-gram shards (Flash: 128 keys, d4-K2048, group 32). Reading only
    the first three left Flash's PLE out of every registry line (2026-09-28)."""
    out = {}
    for blk, kind in (("vq_modules", "expert"), ("vq_linear", "linear"), ("vq_embed", "embed")):
        for m, c in (cfg.get(blk) or {}).items():
            out[m] = {**c, "kind": kind}
    ple = cfg.get("vq_ple") or {}
    g = ple.get("geometry") or {}
    for m in ple.get("keys") or ():
        out[m] = {"dim": g.get("dim"), "k": g.get("k"), "group": g.get("group"), "kind": "ple"}
    return out


def geometry_mix(cfg):
    specs = vq_specs(cfg)
    c = collections.Counter(("ple:" if s["kind"] == "ple" else "")
                            + f"d{s.get('dim')}-K{s.get('k')}" for s in specs.values())
    return dict(sorted(c.items())), len(specs)


def entry(d: pathlib.Path) -> dict:
    cfg = json.loads((d / "config.json").read_text())
    geo, n = geometry_mix(cfg)
    shards = sorted(d.glob("*.safetensors"))
    text_bytes = sum(os.path.getsize(os.path.realpath(f)) for f in shards)
    fp = provenance.shard_fingerprint(d)
    rt = provenance.runtime_state(d)
    rec = None
    if (d / provenance.RECORD).exists():
        rec = json.loads((d / provenance.RECORD).read_text())
    name = d.name
    hf = name.replace(HF_OWNER_SEP, "/", 1) if HF_OWNER_SEP in name else None
    return {
        "name": name, "path": str(d), "host": socket.gethostname().split(".")[0],
        "hf_repo": hf, "model_type": cfg.get("model_type"),
        "vq_modules": n, "geometry": geo,
        "text_gib": round(text_bytes / 2 ** 30, 3), "shards": len(shards),
        "fingerprint": fp,
        "runtime": ({"md5": rt["model_py_md5"][:8], "profile": rt["profile"],
                     "flags": rt["flags"]} if rt else None),
        "build_record": rec["id"] if rec else None,
        "built_by": rec["tool"]["name"] if rec else None,
        "recipe": (rec.get("method") or None) if rec else None,
        "scanned": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }


def read_reg():
    out = {}
    if REG.exists():
        for line in REG.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                out[(e["host"], e["path"])] = e
    return out


def write_reg(reg):
    REG.parent.mkdir(parents=True, exist_ok=True)
    REG.write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in
                           sorted(reg.values(), key=lambda e: (e["name"], e["host"], e["path"]))))


def git_blob_sha1(p):
    data = pathlib.Path(p).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


# Local safety copies left by in-place repairs (model.py.pre-arc6,
# config.json.pre-rows8, model.py.published, *-bak, ...): not drift.
_BACKUP = re.compile(r"\.(pre[-_][\w-]+|published|rows8|[\w-]*bak)$")


def hub_diff(art: pathlib.Path, repo: str, deep=False):
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, files_metadata=True)
    remote = {s.rfilename: s for s in info.siblings}
    rows = []
    local = {f.name: f for f in art.iterdir() if f.is_file()
             and f.name not in provenance.NOT_OUTPUTS and not f.name.startswith(".")}
    for name in sorted(set(local) | set(remote)):
        l, r = local.get(name), remote.get(name)
        if name == ".gitattributes" and r is not None and l is None:
            continue                                  # Hub housekeeping
        if l is not None and r is None and _BACKUP.search(name):
            rows.append((name, "backup", ""))         # local safety copy
            continue
        if l is None:
            rows.append((name, "hub-only", ""))
            continue
        if r is None:
            rows.append((name, "local-only", ""))
            continue
        lp = os.path.realpath(l)
        note = ""
        if r.lfs:
            same = os.path.getsize(lp) == r.lfs.size
            if same and deep:
                same = provenance._sha(lp) == r.lfs.sha256
            elif same:
                note = "size only (--deep hashes)"
        else:
            same = git_blob_sha1(lp) == r.blob_id
        if name == "model.py":
            note = f"local profile {runtime_profile.profile_of(pathlib.Path(lp).read_text()) or 'custom'}"
        rows.append((name, "same" if same else "DIFFERS", note))
    return info.sha, rows


def _save_hub(repo, rev, art, rows, deep):
    """Keep the LATEST comparison per repo (the GUI and `list` read it)."""
    hub = {}
    if HUB.exists():
        for line in HUB.read_text().splitlines():
            if line.strip():
                h = json.loads(line)
                hub[h["repo"]] = h
    hub[repo] = {"repo": repo, "revision": rev, "local": art, "deep": deep,
                 "checked": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                 "drift": [{"file": n, "state": s_, "note": nt} for n, s_, nt in rows
                           if s_ not in ("same", "backup")],
                 "backups": sum(1 for r in rows if r[1] == "backup")}
    HUB.write_text("".join(json.dumps(h, sort_keys=True) + "\n" for h in sorted(
        hub.values(), key=lambda h: h["repo"])))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vqlab registry", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("scan")
    ps.add_argument("roots", nargs="+")
    pl = sub.add_parser("list")
    pl.add_argument("--profile")
    pl.add_argument("--family", help="model_type substring")
    pl.add_argument("--grep")
    ph = sub.add_parser("hub")
    ph.add_argument("artifact", nargs="?")
    ph.add_argument("--repo")
    ph.add_argument("--all", action="store_true")
    ph.add_argument("--deep", action="store_true",
                    help="full sha256 of every LFS shard (reads every byte; slow)")
    a = ap.parse_args(argv)

    if a.cmd == "scan":
        reg = read_reg()
        n = 0
        for root in a.roots:
            root = pathlib.Path(root)
            cands = [root] if is_artifact(root) else [d for d in sorted(root.iterdir())
                                                      if d.is_dir() and is_artifact(d)]
            for d in cands:
                e = entry(d)
                reg[(e["host"], e["path"])] = e
                n += 1
                rt = e["runtime"] or {}
                print(f"  {e['name'][:52]:52s} {e['text_gib']:8.2f} GiB  "
                      f"{rt.get('profile', '-'):7s} {rt.get('md5', '-'):8s}  "
                      f"{'record ' + e['build_record'][:8] if e['build_record'] else 'no record'}")
        write_reg(reg)
        print(f"registered {n} artifacts -> {REG} [{len(reg)} total]")
        return 0

    reg = read_reg()
    if a.cmd == "list":
        for e in reg.values():
            rt = e.get("runtime") or {}
            if a.profile and rt.get("profile") != a.profile:
                continue
            if a.family and a.family not in (e.get("model_type") or ""):
                continue
            if a.grep and a.grep not in e["name"]:
                continue
            print(f"{e['name'][:52]:52s} {e['text_gib']:8.2f} GiB  {rt.get('profile', '-'):7s} "
                  f"{rt.get('md5', '-'):8s}  {e['vq_modules']:4d} mods  "
                  f"{' '.join(f'{g}x{c}' for g, c in e['geometry'].items())[:60]}")
        return 0

    if a.cmd == "hub":
        targets = []
        if a.all:
            targets = [(pathlib.Path(e["path"]), e["hf_repo"]) for e in reg.values()
                       if e.get("hf_repo") and pathlib.Path(e["path"]).is_dir()]
        elif a.artifact:
            art = pathlib.Path(a.artifact)
            repo = a.repo or (art.name.replace(HF_OWNER_SEP, "/", 1)
                              if HF_OWNER_SEP in art.name else None)
            if not repo:
                print("no --repo and the dir name does not encode one (owner--name)")
                return 2
            targets = [(art, repo)]
        rc = 0
        for art, repo in targets:
            try:
                rev, rows = hub_diff(art, repo, a.deep)
            except Exception as e:                      # noqa: BLE001
                print(f"{repo}: could not read the Hub ({type(e).__name__}: {e})")
                rc = max(rc, 1)
                continue
            bad = [r for r in rows if r[1] not in ("same", "backup")]
            _save_hub(repo, rev, str(art), rows, a.deep)
            nb = sum(1 for r in rows if r[1] == "backup")
            if not a.deep:
                print("         (LFS shards compared by size; --deep proves bytes)")
            print(f"{'DRIFT' if bad else 'same '}  {repo}@{rev[:10]}  <- {art}"
                  + (f"  ({nb} local backup files ignored)" if nb else ""))
            for name, st, note in bad:
                print(f"         {st:10s} {name}  {note}")
            rc = max(rc, 2 if bad else 0)
        return rc
    return 2


if __name__ == "__main__":
    sys.exit(main())
