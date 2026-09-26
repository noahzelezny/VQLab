#!/usr/bin/env python3
"""vqlab fits — find, index and file fitted VQ modules (core/fitstore.py).

    vqlab fits index <dir> [<dir> ...]     READ-ONLY: record every module fit
                                           found under the dirs, where it lies
    vqlab fits list [--family F] [--teacher T] [--layers 28-35] [--proj P]
                    [--geom d4-K256] [--recipe-known]
    vqlab fits stats                        counts by family/teacher/geometry
    vqlab fits file [--limit N]             copy indexed fits into the
                                            canonical layout (never deletes
                                            the source); re-verified by fit_id

--root picks the store (default: first of $VQLAB_FIT_STORE, else the
SSD store). Indexing never moves or modifies anything; `file`
copies, verifies, and repoints the index at the filed copy. Removing the
old run dirs afterwards is a separate, human decision.
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # src/
from vqlab import _layout  # noqa: E402,F401  one module object per name
import fitstore as fs  # noqa: E402

FAMILIES = fs._families_dir()   # honours VQLAB_FAMILIES_DIR


def _layers(s):
    if not s:
        return None
    out = set()
    for part in s.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vqlab fits", description=__doc__.split("\n")[0])
    ap.add_argument("--root", help="fit store root (default: first configured)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("index")
    pi.add_argument("dirs", nargs="+")
    pl = sub.add_parser("list")
    for f in ("family", "teacher", "proj", "geom", "layers"):
        pl.add_argument(f"--{f}")
    pl.add_argument("--recipe-known", action="store_true")
    pl.add_argument("--paths", action="store_true", help="print file locations")
    sub.add_parser("stats")
    pf = sub.add_parser("file")
    pf.add_argument("--limit", type=int, default=0)
    pf.add_argument("--family")
    pr = sub.add_parser("retag", help="re-file fits named after a teacher COPY under the "
                        "profiled teacher it is byte-identical to (dry run unless --apply)")
    pr.add_argument("--from", dest="old", required=True, help="teacher name the fits are filed under")
    pr.add_argument("--teacher", required=True,
                    help="that teacher's directory: its config + shard fingerprint must "
                         "match a family profile, which supplies the new name")
    pr.add_argument("--family")
    pr.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)

    root = pathlib.Path(a.root) if a.root else fs.roots()[0]
    root.mkdir(parents=True, exist_ok=True)
    recs = fs.read_index(root)

    if a.cmd == "retag":
        new = fs.teacher_slug(a.teacher, FAMILIES)
        if new == a.old:
            print(f"REFUSE: {a.teacher} matches no profiled teacher by content "
                  f"(config + shard fingerprint); nothing proves {a.old!r} is another teacher")
            return 2
        moves = fs.retag(root, a.old, new, a.family, apply=a.apply)
        print(f"{a.teacher}\n  is byte-identical (config + shard fingerprint) to profiled teacher {new}")
        print(f"{'moved' if a.apply else 'would move'} {len(moves)} fits: {a.old} -> {new}"
              + ("" if a.apply else "   (dry run; --apply to move)"))
        for src, dst in moves[:3]:
            print(f"  {src.relative_to(root)}\n    -> {dst.relative_to(root)}")
        return 0

    if a.cmd == "index":
        sig = fs.load_signatures(FAMILIES)
        new = dup = err = 0
        for r in fs.scan(a.dirs, sig):
            if "_error" in r:
                err += 1
                print("  SKIP", r["_error"][:160])
                continue
            old = recs.get(r["fit_id"])
            if old:
                dup += 1
                old.setdefault("also_at", [])
                if r["location"] not in old["also_at"] and r["location"] != old["location"]:
                    old["also_at"].append(r["location"])
                if not old.get("recipe") and r.get("recipe"):
                    old["recipe"] = r["recipe"]
                continue
            recs[r["fit_id"]] = r
            new += 1
        fs.write_index(root, recs)
        print(f"indexed {new} new fits ({dup} already known, {err} unreadable) "
              f"-> {fs.index_path(root)}  [{len(recs)} total]")
        return 0

    if a.cmd == "stats":
        c = collections.Counter((r["family"], r["teacher"]) for r in recs.values())
        g = collections.Counter((r["family"], f"d{r['d']}-K{r['K']}") for r in recs.values())
        filed = sum(1 for r in recs.values() if r.get("sha256"))
        known = sum(1 for r in recs.values() if r.get("recipe"))
        print(f"{len(recs)} fits in {fs.index_path(root)}: {filed} filed in the "
              f"canonical layout, {known} with a known recipe")
        for (fam, t), n in sorted(c.items()):
            print(f"  {n:6d}  {fam}/{t}")
        print("  by geometry:")
        for (fam, geo), n in sorted(g.items()):
            print(f"  {n:6d}  {fam:14s} {geo}")
        return 0

    if a.cmd == "list":
        d = K = None
        if a.geom:
            ds, _, ks = a.geom.partition("-K")
            d, K = int(ds.lstrip("d")), int(ks)
        hits = fs.query(recs, a.family, a.teacher, _layers(a.layers), a.proj, d, K,
                        True if a.recipe_known else None)
        for r in hits:
            rc = r.get("recipe") or {}
            how = (f"{rc.get('origin')}"
                   f"{' alt' if (rc.get('fitter') or {}).get('alternation') else ''}"
                   f"{' seed=' + str((rc.get('fitter') or {}).get('seed')) if rc.get('fitter') else ''}"
                   if rc else f"recipe? ({r['source']['run_dir']})")
            print(f"{r['family']}/{r['teacher']}  L{r.get('layer')} {r['proj']:10s} "
                  f"d{r['d']}-K{r['K']}  {r['fit_id']}  {how}"
                  + (f"\n    {r['location']}" if a.paths else ""))
        print(f"{len(hits)} fits")
        return 0

    if a.cmd == "file":
        todo = [r for r in recs.values() if not r.get("sha256")
                and r["family"] != "_unattributed"
                and (not a.family or r["family"] == a.family)]
        if a.limit:
            todo = todo[: a.limit]
        done = 0
        for r in todo:
            recs[r["fit_id"]] = fs.file_fit(root, r)
            done += 1
            if done % 100 == 0:
                fs.write_index(root, recs)
                print(f"  filed {done}/{len(todo)}", flush=True)
        fs.write_index(root, recs)
        skipped = sum(1 for r in recs.values() if r["family"] == "_unattributed")
        print(f"filed {done} fits under {root / 'fits'} "
              f"({skipped} unattributed left in place; nothing deleted)")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
