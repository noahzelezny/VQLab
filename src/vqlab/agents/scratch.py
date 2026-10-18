#!/usr/bin/env python3
"""vqlab scratch: what on the scratch volume a human could delete, and why.

    vqlab scratch reclaimable [--root DIR ...] [--full] [--json]

Surveys the configured scratch root (`vqlab.config`), or each `--root` (any
folder, HDD ones too), and prints, per class, what is provably redundant with
its evidence, sizes and a grand total, then the `rm -rf` lines. It never
deletes anything: deletion is the human's action, and there is no --delete
flag. Everything else at the top of a root is reported KEPT with the reason,
so an absence is never silent.

The SSD went from 482 GiB free to 15 GiB in one night because unpacked fits
sat beside their packed copies (lab/docs/OPERATOR-NOTES-2026-10-03.md, sec 3),
and on 2026-10-17 this command found 2 GiB where a manual inventory found
~1.1 TiB (lab/docs/STORAGE-INVENTORY-2026-10-17.md). The classes:

  packed     an unpacked dir whose PACKED counterpart's build record (`vqlab
             pack` / `pack-dense`) names it, when the pack still matches its
             record (sizes; `--full` re-hashes every byte), the source still
             IS what was packed (record id + shard fingerprint), and nothing
             links into it.
  preflight  a queue preflight's copies (`config.preflight_dir()/<queue>`)
             whose queue is finished (passed, failed, stopped, deferred, or a
             `running` one whose runner died) and has no live runner; or whose
             queue record no longer exists. A running queue's preflight is KEPT.
  fits       a dir every one of whose safetensors files holds only VQ fit
             triples, each already FILED in the fit store: its fit_id and size
             in the `fit_store` index (cheap), then every tensor byte equal to
             the filed copy (read only when the cheap checks pass; `--full`
             also re-hashes the filed copy against its recorded sha256). One
             unfiled or differing module, or one stray tensor, keeps it.
  caches     a KL teacher cache dir (teacher_topk.safetensors) byte-identical
             (same files, sizes, sha256) to a REGISTERED cache elsewhere
             (families/**/teacher_caches.json). A registered cache is ALWAYS
             kept: the release gate is measured against it.
  campaigns  a top-level `F###-name` folder whose finding has an entry in the
             findings log: an ARCHIVE candidate (finished, possibly unique),
             never on the rm list.

Anything a symlink (under the roots or the models root) resolves into is kept:
a pin, a minibase, a mix or a served name would break when it goes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shlex
import sys
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field

from vqlab import config as C
from vqlab.agents.doctor import CAMPAIGN
from vqlab.core.source import fit_store
from vqlab.family import registry
from vqlab.ops.findings import findings_entries
from vqlab.queue import report as queue_report
from vqlab.queue.files import load_state
from vqlab.records import provenance
from vqlab.util import hashing
from vqlab.util.fs import GIB, dir_bytes

# tool name in a build record -> the input role that names what it packed
PACKERS = {"pack": "src", "pack-dense": "src"}
MAX_DEPTH = 4
CACHE_MARK = "teacher_topk.safetensors"     # kl_damage writes it into every cache
CLASSES = ("packed", "preflight", "fits", "caches", "campaigns")
DELETE_CLASSES = ("packed", "preflight", "fits", "caches")


@dataclass(frozen=True)
class Item:
    """One listed dir: its class, size and the evidence that makes it safe."""
    cls: str
    path: str
    bytes: int
    evidence: str
    detail: dict = field(default_factory=dict)   # class-specific facts, e.g. the pack


@dataclass(frozen=True)
class Kept:
    """One dir a class looked at and refused, with the reason."""
    cls: str
    path: str
    reason: str


def _under(p: str, roots: Iterable[str]) -> bool:
    return any(p == r or p.startswith(r + os.sep) for r in roots)


def _walk(roots: Iterable[pathlib.Path], depth: int = MAX_DEPTH) -> Iterator[pathlib.Path]:
    """Every dir under `roots` down to `depth`, hidden dirs skipped, each
    real dir once."""
    seen: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for dirpath, dirnames, _ in os.walk(root):
            d = pathlib.Path(dirpath)
            level = len(d.relative_to(root).parts)
            dirnames[:] = [n for n in dirnames if not n.startswith(".")] if level < depth else []
            rp = os.path.realpath(d)
            if rp not in seen:
                seen.add(rp)
                yield d


def _symlinks(roots: Iterable[pathlib.Path]) -> list[tuple[str, str]]:
    """(link, resolved target) for every symlink under `roots`, in one pass."""
    out = []
    for root in roots:
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames + dirnames:
                p = pathlib.Path(dirpath, name)
                if p.is_symlink():
                    out.append((str(p), os.path.realpath(p)))
    return out


def _links_into(links: list[tuple[str, str]], target: pathlib.Path) -> list[str]:
    """Links (outside `target` itself) that resolve into it."""
    t = os.path.realpath(target)
    return [p for p, rp in links if _under(rp, [t]) and not _under(os.path.realpath(
        pathlib.Path(p).parent), [t])]


# ------------------------------------------------------------------- packed
def _packed_ok(art: pathlib.Path, rec: dict, full: bool) -> str | None:
    """None if the packed artifact still matches its record, else why not."""
    if full:
        bad = provenance.verify(art, rec)
        return f"{bad[0][0]}: {bad[0][1]}" if bad else None
    if hashlib.sha256(provenance.canon(rec).encode()).hexdigest() != rec.get("id"):
        return "record edited after it was written (id mismatch)"
    for name, want in rec.get("outputs", {}).items():
        f = art / name
        if name in provenance.JUNK or name.startswith("._"):
            continue
        if not f.exists():
            return f"{name}: missing"
        if f.resolve().stat().st_size != want["bytes"]:
            return f"{name}: bytes differ from its record"
    return None


def _packed_sources(roots: list[pathlib.Path]) -> Iterator[tuple[pathlib.Path, dict, dict]]:
    """(packed artifact, its record, the input it packed) under `roots`."""
    for art in _walk(roots):
        if not (art / provenance.RECORD).is_file():
            continue
        try:
            rec = provenance.load(art)
        except (OSError, ValueError):
            continue
        role = PACKERS.get((rec.get("tool") or {}).get("name"))
        for inp in rec.get("inputs", []) if role else ():
            if inp.get("role") == role:
                yield art, rec, inp


def packed_items(roots: list[pathlib.Path], links: list[tuple[str, str]],
                 full: bool = False) -> tuple[list[Item], list[Kept]]:
    """Unpacked dirs whose verified pack exists (class `packed`)."""
    rroots = [os.path.realpath(r) for r in roots]
    items, kept, claimed = [], [], set()
    for art, rec, inp in _packed_sources(roots):
        tool = rec["tool"]["name"]
        src = pathlib.Path(inp.get("realpath") or inp.get("path", ""))
        rsrc = os.path.realpath(src)
        if not _under(rsrc, rroots) or rsrc in claimed or rsrc == os.path.realpath(art) \
                or not src.is_dir():
            continue
        why = _packed_ok(art, rec, full)
        have = None if why else hashing.fingerprint(src)
        pid = inp.get("provenance_id")
        if not why and pid and have.get("provenance_id") != pid:
            why = "source's build record changed since it was packed"
        elif not why and have.get("shards") != inp.get("shards"):
            why = "source shards changed since they were packed"
        elif why:
            why = f"packed artifact {art} does not verify ({why})"
        if not why and (hit := _links_into(links, src)):
            why = f"{len(hit)} symlink(s) resolve into it, e.g. {hit[0]}"
        if why:
            kept.append(Kept("packed", str(src), why))
            continue
        claimed.add(rsrc)
        items.append(Item("packed", str(src), dir_bytes(src),
                          f"packed -> {art} ({tool}, record {rec['id'][:12]})",
                          {"packed": str(art), "tool": tool, "record": rec["id"]}))
    return items, kept


def reclaimable(roots: Iterable, full: bool = False) -> tuple[list[dict], list[dict]]:
    """The `packed` class alone, as the GUI reads it: (candidates [{path, bytes,
    packed, tool, record}], kept [{path, reason}])."""
    roots = [pathlib.Path(r) for r in roots]
    items, kept = packed_items(roots, _symlinks(roots), full)
    cands = [{"path": it.path, "bytes": it.bytes, **it.detail} for it in items]
    return cands, [{"path": k.path, "reason": k.reason} for k in kept]


# ---------------------------------------------------------------- preflight
def _preflight_roots(roots: list[pathlib.Path]) -> list[pathlib.Path]:
    """The configured preflight dir and any same-named dir at a root's top."""
    name = C.preflight_dir().name
    cands = [C.preflight_dir(), *(r / name for r in roots)]
    rroots = [os.path.realpath(r) for r in roots]
    out, seen = [], set()
    for p in cands:
        rp = os.path.realpath(p)
        if p.is_dir() and _under(rp, rroots) and rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def _queue_verdict(qdir: pathlib.Path) -> tuple[bool, str]:
    """(finished, evidence) for the queue a preflight folder belongs to."""
    if not (qdir / "state.json").is_file():
        return True, (f"its queue record {qdir} does not exist (a queue run from another "
                      "box keeps its record on that box)")
    state = load_state(qdir)
    status = queue_report.live_status(qdir, state)
    pid = queue_report.runner_pid(qdir, state, strict=False)
    if pid is not None:
        return False, f"queue {qdir.name} has a live runner (pid {pid})"
    if status in (*queue_report.TERMINAL, "died"):
        return True, f"queue {qdir.name} is {status}, no live runner"
    return False, f"queue {qdir.name} is {status}"


def preflight_items(roots: list[pathlib.Path], links: list[tuple[str, str]]
                    ) -> tuple[list[Item], list[Kept]]:
    """Preflight copies whose queue has finished (class `preflight`)."""
    items, kept = [], []
    for pf in _preflight_roots(roots):
        for d in sorted(p for p in pf.iterdir() if p.is_dir() and not p.name.startswith(".")):
            done, why = _queue_verdict(C.queues_dir() / d.name)
            if done and (hit := _links_into(links, d)):
                done, why = False, f"{len(hit)} symlink(s) resolve into it, e.g. {hit[0]}"
            if done:
                items.append(Item("preflight", str(d), dir_bytes(d), why))
            else:
                kept.append(Kept("preflight", str(d), why))
    return items, kept


# --------------------------------------------------------------------- fits
def fit_index() -> dict[str, dict]:
    """Every FILED fit (a record with a sha256) across the fit-store roots."""
    out: dict[str, dict] = {}
    for root in fit_store.roots():
        out.update({k: r for k, r in fit_store.read_index(root).items() if r.get("sha256")})
    return out


def _same_bytes(src: pathlib.Path, h: dict, base: int, rec: dict, module: str,
                full: bool) -> str | None:
    """None when `src`'s copy of `module` holds the same tensor bytes as the
    filed copy (and, with `full`, the filed copy still hashes to its recorded
    sha256), else what differs. fit_id hashes the codebook only, so a refit
    of the codes under the same codebook shares it: the bytes decide."""
    filed = pathlib.Path(rec["location"])
    if full and hashing.sha256_file(filed) != rec["sha256"]:
        return f"filed copy {filed} no longer matches its recorded sha256"
    fh, fbase = fit_store.read_header(filed)
    for key in (module + s for s in fit_store.VQ_SUFFIXES):
        if fit_store.read_tensor_bytes(src, h, base, key) != \
                fit_store.read_tensor_bytes(filed, fh, fbase, key):
            return f"{key} differs from the filed copy {filed} (same fit_id)"
    return None


def _file_verdict(f: pathlib.Path, index: dict[str, dict], here: str,
                  full: bool) -> tuple[int, str | None]:
    """(fit modules in the file, why it is not provably filed) for one
    safetensors file of a candidate fit dir."""
    try:
        h, base = fit_store.read_header(f)
    except Exception as e:  # truncated or not safetensors: not provably anything
        return 0, f"{f.name}: unreadable ({e})"
    mods = fit_store.modules_in(h)
    triple = {m + s for m in mods for s in fit_store.VQ_SUFFIXES}
    other = sorted(set(h) - triple - {"__metadata__"})
    if other:
        # an artifact's shard, not a fit part: counts as no fit (n stays 0 if first)
        return 0, f"{f.name} holds {len(other)} non-fit tensor(s), e.g. {other[0]}"
    why = next((w for m in mods if (w := _module_verdict(f, h, base, m, index, here, full))),
               None)
    return len(mods), why


def _module_verdict(f: pathlib.Path, h: dict, base: int, m: str, index: dict[str, dict],
                    here: str, full: bool) -> str | None:
    """None when module `m` of `f` is provably filed elsewhere, else why not."""
    d = fit_store.describe(f, h, base, m)
    rec = index.get(d["fit_id"])
    if rec is None:
        return f"{f.name}: {m} is not filed in the fit store (fit_id {d['fit_id']})"
    loc = pathlib.Path(rec["location"])
    if rec.get("bytes") != d["bytes"] or not loc.is_file():
        return f"{f.name}: {m}'s filed copy {loc} is missing or a different size"
    if _under(os.path.realpath(loc), [here]):
        return f"{f.name}: {m}'s only filed copy is inside this dir"
    return _same_bytes(f, h, base, rec, m, full)


def _fit_dirs(roots: list[pathlib.Path], skip: list[str]) -> Iterator[pathlib.Path]:
    """Dirs under `roots` holding safetensors directly and no subfolders, outside
    `skip` (the store's own fits/ tree, preflights, caches)."""
    for d in _walk(roots):
        if _under(os.path.realpath(d), skip) or (d / CACHE_MARK).exists():
            continue
        entries = [p for p in d.iterdir() if not p.name.startswith(".")]
        if any(p.suffix == ".safetensors" for p in entries) and \
                not any(p.is_dir() for p in entries):
            yield d


def fit_items(roots: list[pathlib.Path], links: list[tuple[str, str]], full: bool = False,
              index: dict[str, dict] | None = None) -> tuple[list[Item], list[Kept]]:
    """Fit dirs whose every module is already filed (class `fits`)."""
    index = fit_index() if index is None else index
    skip = [os.path.realpath(r / "fits") for r in fit_store.roots()]
    skip += [os.path.realpath(p) for p in _preflight_roots(roots)]
    items, kept = [], []
    for d in _fit_dirs(roots, skip):
        here = os.path.realpath(d)
        tensors = sorted(d.glob("*.safetensors"))
        n, why = 0, None
        for f in tensors:
            got, why = _file_verdict(f, index, here, full)
            n += got
            if why:
                break
        if n == 0:
            continue                                    # no fit triple seen: not a fit dir
        if not why and (hit := _links_into(links, d)):
            why = f"{len(hit)} symlink(s) resolve into it, e.g. {hit[0]}"
        if why:
            kept.append(Kept("fits", str(d), why))
            continue
        side = sorted(p.name for p in d.iterdir()
                      if p.suffix != ".safetensors" and p.name not in provenance.JUNK)
        level = "fit_id, tensor bytes + filed sha256" if full else "fit_id + tensor bytes"
        items.append(Item("fits", str(d), dir_bytes(d),
                          f"{n}/{n} modules in {len(tensors)} file(s) filed in the fit store "
                          f"({level})" + (f"; sidecars {', '.join(side)}" if side else "")))
    return items, kept


# ------------------------------------------------------------------- caches
def _cache_files(d: pathlib.Path) -> dict[str, int]:
    """{relative name: bytes} of a cache dir's real files, junk skipped."""
    return {str(p.relative_to(d)): p.stat().st_size for p in sorted(d.rglob("*"))
            if p.is_file() and p.name not in provenance.JUNK and not p.name.startswith("._")}


def _identical(a: pathlib.Path, b: pathlib.Path, files: dict[str, int]) -> bool:
    return all(hashing.sha256_file(a / n) == hashing.sha256_file(b / n) for n in files)


def cache_items(roots: list[pathlib.Path], links: list[tuple[str, str]],
                caches: list[dict] | None = None) -> tuple[list[Item], list[Kept]]:
    """Unregistered cache dirs byte-identical to a registered one (class `caches`)."""
    caches = registry.teacher_caches() if caches is None else caches
    reg = {os.path.realpath(C.expand(c.get("path") or "")): c for c in caches}
    reg = {p: c for p, c in reg.items() if pathlib.Path(p).is_dir()}
    sigs: dict[str, dict[str, int]] = {}
    items, kept = [], []
    for d in _walk(roots):
        if not (d / CACHE_MARK).is_file():
            continue
        rp = os.path.realpath(d)
        if rp in reg:
            c = reg[rp]
            kept.append(Kept("caches", str(d), f"registered cache ({c['teacher']} "
                                               f"{c.get('corpus')}): never deleted"))
            continue
        files = _cache_files(d)
        twin = None
        for p in reg:
            if sigs.setdefault(p, _cache_files(pathlib.Path(p))) == files \
                    and _identical(d, pathlib.Path(p), files):
                twin = p
                break
        why = None if twin else ("not registered, and no registered cache holds the same "
                                 "bytes (register it, or it is its only copy)")
        if twin and (hit := _links_into(links, d)):
            why = f"{len(hit)} symlink(s) resolve into it, e.g. {hit[0]}"
        if why:
            kept.append(Kept("caches", str(d), why))
            continue
        c = reg[twin]
        items.append(Item("caches", str(d), dir_bytes(d),
                          f"{len(files)} file(s) sha256-identical to the registered "
                          f"{c['teacher']} {c.get('corpus')} cache {twin}"))
    return items, kept


# ---------------------------------------------------------------- campaigns
def campaign_items(roots: list[pathlib.Path], entries: list[dict] | None = None
                   ) -> tuple[list[Item], list[Kept]]:
    """Top-level F###-name folders whose finding is written (class `campaigns`,
    ARCHIVE candidates, never deleted)."""
    entries = findings_entries(C.findings_log()) if entries is None else entries
    by_f = {f"F{e['f']}": e for e in entries}
    items, kept = [], []
    for root in roots:
        for d in sorted(root.iterdir()) if root.is_dir() else ():
            m = CAMPAIGN.match(d.name)
            if not (m and d.is_dir()):
                continue
            e = by_f.get(f"F{int(m.group(1)[1:])}")
            if e:
                items.append(Item("campaigns", str(d), dir_bytes(d),
                                  f"findings log line {e['line']}: {e['headline'][:90]}"))
            else:
                kept.append(Kept("campaigns", str(d), f"{m.group(1)} has no entry in the "
                                 "findings log yet: this folder may be its only record"))
    return items, kept


# ------------------------------------------------------------------- survey
def _nested_out(items: list[Item]) -> list[Item]:
    """Drop an item inside another listed item (its bytes are counted once)."""
    paths = [os.path.realpath(i.path) for i in items]
    return [i for i, p in zip(items, paths, strict=True)
            if not any(q != p and _under(p, [q]) for q in paths)]


def _top_level_kept(roots: list[pathlib.Path], items: list[Item], kept: list[Kept]) -> list[Kept]:
    """Every top-level dir of a root no class listed, with the reason."""
    listed = [os.path.realpath(i.path) for i in items]
    refused = {os.path.realpath(k.path) for k in kept}
    out = []
    for root in roots:
        if os.path.realpath(root) in listed:     # the root itself is a listed item
            continue
        entries = sorted(root.iterdir()) if root.is_dir() else []
        loose = [p for p in entries if not p.is_dir() and not p.name.startswith(".")]
        if loose:
            out.append(Kept("unclassified", str(root), f"{len(loose)} loose file(s) at the top, "
                            f"e.g. {loose[0].name}: no class covers loose files"))
        for d in entries:
            rp = os.path.realpath(d)
            if not d.is_dir() or d.name.startswith(".") or rp in refused or rp in listed:
                continue
            inside = sum(_under(p, [rp]) for p in listed)
            out.append(Kept("unclassified", str(d), f"partly listed ({inside} item(s) inside); "
                            "the rest is proved redundant by no class" if inside else
                            "no class proves it redundant"))
    return out


def survey(roots: Iterable, full: bool = False) -> dict:
    """Every class over `roots`: {roots, items, kept, total_bytes}."""
    roots = [pathlib.Path(r) for r in roots]
    links = _symlinks([*roots, C.models()])
    items: list[Item] = []
    kept: list[Kept] = []
    for got, refused in (packed_items(roots, links, full), preflight_items(roots, links),
                         fit_items(roots, links, full), cache_items(roots, links),
                         campaign_items(roots)):
        items += got
        kept += refused
    items = _nested_out(items)
    kept += _top_level_kept(roots, items, kept)
    return {"roots": [str(r) for r in roots], "items": [asdict(i) for i in items],
            "kept": [asdict(k) for k in kept],
            "total_bytes": sum(i.bytes for i in items if i.cls in DELETE_CLASSES),
            "archive_bytes": sum(i.bytes for i in items if i.cls not in DELETE_CLASSES)}


def _print(rep: dict) -> None:
    print(f"roots: {', '.join(rep['roots'])}")
    for cls in CLASSES:
        rows = sorted((i for i in rep["items"] if i["cls"] == cls), key=lambda i: -i["bytes"])
        word = "ARCHIVE candidates" if cls not in DELETE_CLASSES else "reclaimable"
        print(f"\n== {cls}: {sum(i['bytes'] for i in rows) / GIB:.2f} GiB {word} "
              f"in {len(rows)} dir(s)")
        for i in rows:
            print(f"  {i['bytes'] / GIB:9.2f} GiB  {i['path']}\n               {i['evidence']}")
    print("\n== KEPT")
    for k in rep["kept"]:
        print(f"  KEPT [{k['cls']}] {k['path']}\n        {k['reason']}")
    dels = [i for i in rep["items"] if i["cls"] in DELETE_CLASSES]
    print(f"\ntotal {rep['total_bytes'] / GIB:.2f} GiB reclaimable in {len(dels)} dir(s); "
          f"{rep['archive_bytes'] / GIB:.2f} GiB of finished campaigns to archive")
    if dels:
        print("\n# Nothing was deleted. To reclaim the space, a human runs:")
        for i in dels:
            print(f"rm -rf {shlex.quote(i['path'])}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vqlab scratch", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reclaimable", help="list what is provably redundant, with evidence; "
                                           "print the rm lines")
    r.add_argument("--root", action="append", default=None, type=pathlib.Path,
                   help="survey this dir instead of the configured scratch (repeatable)")
    r.add_argument("--full", action="store_true",
                   help="re-hash packed outputs and filed fit copies against their "
                        "records (default: record ids and sizes)")
    r.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    rep = survey(a.root or [C.scratch()], full=a.full)
    if a.json:
        print(json.dumps(rep, indent=1))
    else:
        _print(rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
