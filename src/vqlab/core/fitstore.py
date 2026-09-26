"""The fit store: every fitted VQ module, filed by WHAT it is, not which run made it.

A k-means fit is the expensive thing in this lab (~20 GPU-min per 397B
layer), and mixed-codebook builds are assembled from fits at DIFFERENT
geometries per module. So fits are organized broad -> specific:

    <root>/fits/<family>/<teacher>/L<layer>/<proj>/d<D>-K<K>/<fit_id>.safetensors
                                                             <fit_id>.json   recipe + identity
    <root>/index.jsonl                                       one line per fit

The same layout on every root (SSD = hot, HDD = cold archive). Roots come
from $VQLAB_FIT_STORE (colon-separated, first = write target), defaulting to
the HDD archive (else ~/.vqlab/fits). One fit = ONE module's codebook + codes +
vq_scales; files that held several modules are split when filed.

Identity. `fit_id` hashes the codebook tensor bytes plus every tensor's
shape and dtype -- a few KB read per fit, where a full hash of the 524 GB
archive would take an hour off the HDD. Two stochastic fits of the same
module never share a codebook, so this separates them; `sha256` of the full
file is recorded whenever a fit is written.

Attribution. A part's (E, OUT, IN) comes from its tensor shapes; the
teacher is the one whose family profile (families/*/teachers/*/profile.json)
has that projection signature. A fit from ANOTHER model at the same (d, K)
-- the 240 35B parts sitting in the 2026-09-15 Flash pool -- therefore files
under its own teacher, not Flash's.

Recipe. How a fit was made (fitter, init, seed, alternation, commit) comes
from geo-build's origins.json when the fit has one; older fits are filed
with `recipe: null` and the name of the run dir they came from, rather than
a guess.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import struct

GSZ = 64
# HDD first: fits are the archive of record (Noah, 2026-09-05: fitted
# tensors go to the HDD so no k-means is paid twice). An SSD root with the
# same layout can be listed in $VQLAB_FIT_STORE as a hot mirror.
_DEFAULT_ROOTS = ("<fits>",)
_MOD_RX = re.compile(r"layers\.(\d+)\..*?([A-Za-z0-9_]+)$")
_SUFFIXES = (".codebook", ".codes", ".vq_scales")


def roots():
    """Configured store roots; the first is where new fits are written.
    Without $VQLAB_FIT_STORE: the lab's storage array stores if mounted, else
    ~/.vqlab/fits -- so a user without the lab's disks still gets a store."""
    env = os.environ.get("VQLAB_FIT_STORE")
    if env:
        return [pathlib.Path(p) for p in env.split(":") if p]
    mounted = [pathlib.Path(p) for p in _DEFAULT_ROOTS if pathlib.Path(p).parent.is_dir()]
    return mounted or [pathlib.Path.home() / ".vqlab" / "fits"]


def _families_dir():
    return pathlib.Path(os.environ.get("VQLAB_FAMILIES_DIR")
                        or pathlib.Path(__file__).resolve().parents[3] / "families")


def teacher_identity(teacher_dir):
    """(config_sha256, shard_fingerprint) -- the same two hashes a family
    profile records under `identity` (per-shard bytes + first-MiB sha256)."""
    from vqlab.records import provenance
    rp = pathlib.Path(os.path.realpath(teacher_dir))
    fp = provenance.fingerprint(rp)
    return (hashlib.sha256((rp / "config.json").read_bytes()).hexdigest(),
            hashlib.sha256(json.dumps(fp.get("shards", {}), sort_keys=True).encode()).hexdigest())


_SLUGS = {}


def teacher_slug(teacher_dir, families_dir=None) -> str:
    """The teacher's name in the store and in families/*/teachers/.

    By CONTENT first: a teacher whose config + shard fingerprint match a
    profiled teacher gets that teacher's name, whatever directory it sits in
    (a scratch copy `paper_rev/teacher_35b_bf16`, byte-identical to the 35B
    teacher, filed 602 fits under its dir name on 2026-09-26 -- VL4.11).
    Else its HF repo when it is a hub snapshot, else its directory name."""
    rp = pathlib.Path(os.path.realpath(teacher_dir))
    fam = pathlib.Path(families_dir) if families_dir else _families_dir()
    cst = (rp / "config.json").stat() if (rp / "config.json").is_file() else None
    key = (str(rp), str(fam), cst and (cst.st_mtime_ns, cst.st_size))
    if key in _SLUGS:
        return _SLUGS[key]
    parts = rp.parts
    if "snapshots" in parts:
        name = parts[parts.index("snapshots") - 1].removeprefix("models--")
    else:
        name = re.sub(r"[^A-Za-z0-9._-]+", "-", rp.name)
    cfg = rp / "config.json"
    if cfg.is_file():
        csha = hashlib.sha256(cfg.read_bytes()).hexdigest()
        cands = []
        for prof in sorted(fam.glob("*/teachers/*/profile.json")):
            ident = json.loads(prof.read_text()).get("identity") or {}
            if ident.get("config_sha256") == csha and ident.get("shard_fingerprint"):
                cands.append((prof.parent.name, ident["shard_fingerprint"]))
        if cands:                                   # config matches: prove the weights
            shard_fp = teacher_identity(rp)[1]
            hits = sorted({t for t, f in cands if f == shard_fp})
            if len(hits) == 1:
                name = hits[0]
    _SLUGS[key] = name
    return name


def retag(root, old, new, family=None, apply=False):
    """Move every fit filed under teacher `old` to teacher `new` (files,
    sidecars, index). Dry run unless apply. Returns [(src, dst)]."""
    root = pathlib.Path(root)
    recs = read_index(root)
    todo = [r for r in recs.values()
            if r["teacher"] == old and (not family or r["family"] == family)]
    clash = [root / rel_path(dict(r, teacher=new)) for r in todo
             if (root / rel_path(dict(r, teacher=new))).with_suffix(".safetensors").exists()]
    if clash:                                     # checked BEFORE the first move
        raise SystemExit(f"refusing: {len(clash)} destinations already exist, e.g. {clash[0]}")
    moves = []
    for r in todo:
        src = root / rel_path(r)
        r2 = dict(r, teacher=new)
        dst = root / rel_path(r2)
        moves.append((src, dst))
        if not apply:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        for ext in (".safetensors", ".json"):
            s_, d_ = src.with_suffix(ext), dst.with_suffix(ext)
            if d_.exists():
                raise SystemExit(f"refusing to overwrite {d_}")
            if s_.exists():
                os.replace(s_, d_)
        if r.get("location") == str(src.with_suffix(".safetensors")):
            r2["location"] = str(dst.with_suffix(".safetensors"))
        r2.setdefault("retagged", []).append(
            {"from": old, "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")})
        dst.with_suffix(".json").write_text(json.dumps(r2, indent=1, sort_keys=True))
        recs[r["fit_id"]] = r2
    if apply and moves:
        write_index(root, recs)
        for src, _ in moves:                          # prune emptied dirs
            d = src.parent
            while d != root and d.is_dir() and not any(d.iterdir()):
                d.rmdir()
                d = d.parent
    return moves


# ------------------------------------------------------------ safetensors I/O
def read_header(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def read_tensor_bytes(p, h, base, key):
    a, b = h[key]["data_offsets"]
    with open(p, "rb") as f:
        f.seek(base + a)
        return f.read(b - a)


def modules_in(h):
    """Module names that carry a full (codebook, codes, vq_scales) triple."""
    mods = {k[: -len(s)] for k in h for s in _SUFFIXES if k.endswith(s)}
    return sorted(m for m in mods if all(m + s in h for s in _SUFFIXES))


# ---------------------------------------------------------------- identity
def describe(p, h, base, module):
    """Everything provable about one module's fit from its header + codebook."""
    cb, codes, sc = (h[module + s] for s in _SUFFIXES)
    K, d = cb["shape"]
    E, OUT = codes["shape"][:2]
    IN = sc["shape"][2] * GSZ
    packed = codes["dtype"] == "U32"
    bits = math.ceil(math.log2(K))
    ident = hashlib.sha256()
    ident.update(read_tensor_bytes(p, h, base, module + ".codebook"))
    ident.update(json.dumps({s: [h[module + s]["shape"], h[module + s]["dtype"]]
                             for s in _SUFFIXES}, sort_keys=True).encode())
    m = _MOD_RX.search(module)
    return {"fit_id": ident.hexdigest()[:16], "module": module,
            "layer": int(m.group(1)) if m else None,
            "proj": m.group(2) if m else module.rsplit(".", 1)[-1],
            "d": int(d), "K": int(K), "E": int(E), "OUT": int(OUT), "IN": int(IN),
            "pack_bits": bits if packed else None,
            "bytes": sum(h[module + s]["data_offsets"][1] - h[module + s]["data_offsets"][0]
                         for s in _SUFFIXES)}


# ------------------------------------------------------------- attribution
def load_signatures(families_dir):
    """{(proj, E, OUT, IN): [(family, teacher), ...]} from every profile."""
    sig = {}
    for prof in pathlib.Path(families_dir).glob("*/teachers/*/profile.json"):
        pj = json.loads(prof.read_text())
        for proj, ss in (pj.get("modules", {}).get("signatures") or {}).items():
            for E, OUT, IN in ss:
                sig.setdefault((proj, E, OUT, IN), []).append(
                    (pj["family"], pj["teacher"]))
    return sig


def attribute(rec, sig):
    hits = sig.get((rec["proj"], rec["E"], rec["OUT"], rec["IN"]), [])
    if len(hits) == 1:
        return hits[0]
    return ("_unattributed", "ambiguous" if hits else "no-matching-profile")


def rel_path(rec):
    L = f"L{rec['layer']:03d}" if rec.get("layer") is not None else "L___"
    return pathlib.Path("fits", rec["family"], rec["teacher"], L, rec["proj"],
                        f"d{rec['d']}-K{rec['K']}", rec["fit_id"])


# ------------------------------------------------------------------- index
def index_path(root):
    return pathlib.Path(root) / "index.jsonl"


def append_index(root, rec):
    """One O_APPEND write: concurrent writers never interleave a line. Later
    lines win in read_index, so an append is also an update."""
    line = (json.dumps(rec, sort_keys=True) + "\n").encode()
    fd = os.open(index_path(root), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def reindex(root):
    """Rebuild the index from the per-fit .json sidecars, which are the
    source of truth for FILED fits (in-place entries are kept as they are)."""
    recs = {r["fit_id"]: r for r in read_index(root).values() if not r.get("sha256")}
    for j in (pathlib.Path(root) / "fits").rglob("*.json"):
        r = json.loads(j.read_text())
        recs[r["fit_id"]] = r
    write_index(root, recs)
    return recs


def put(part_file, module, family, teacher, recipe=None, root=None):
    """File one module's fit from `part_file` into the store and index it.
    Called by every fitter right after it saves a fit. Returns the record."""
    root = pathlib.Path(root) if root else roots()[0]
    root.mkdir(parents=True, exist_ok=True)
    h, base = read_header(part_file)
    rec = describe(part_file, h, base, module)
    rec.update(family=family, teacher=teacher, recipe=recipe,
               source={"file": str(part_file), "run_dir": pathlib.Path(part_file).parent.name,
                       "multi_module": len(modules_in(h)) > 1},
               location=str(part_file))
    rec = file_fit(root, rec)
    append_index(root, rec)
    return rec


def read_index(root):
    p = index_path(root)
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["fit_id"]] = r
    return out


def write_index(root, recs):
    p = index_path(root)
    tmp = p.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, sort_keys=True) + "\n"
                           for r in sorted(recs.values(), key=lambda r: (
                               r["family"], r["teacher"], r.get("layer") or -1,
                               r["proj"], r["d"], r["K"], r["fit_id"]))))
    os.replace(tmp, p)


def recipe_for(part_file, module):
    """geo-build's origins.json ledger, if the part's dir kept one."""
    op = pathlib.Path(part_file).parent / "origins.json"
    if op.exists():
        rec = json.loads(op.read_text()).get(pathlib.Path(part_file).name)
        if rec:
            return rec
    return None


def scan(paths, sig):
    """Yield one record per module fit found under `paths` (read-only)."""
    for top in paths:
        top = pathlib.Path(top)
        files = [top] if top.is_file() else sorted(top.rglob("*.safetensors"))
        for f in files:
            try:
                h, base = read_header(f)
            except Exception as e:                   # truncated / not safetensors
                yield {"_error": f"{f}: {e}"}
                continue
            for mod in modules_in(h):
                rec = describe(f, h, base, mod)
                rec["family"], rec["teacher"] = attribute(rec, sig)
                rec["recipe"] = recipe_for(f, mod)
                rec["source"] = {"file": str(f), "run_dir": f.parent.name,
                                 "multi_module": len(modules_in(h)) > 1}
                rec["location"] = str(f)
                yield rec


def file_fit(root, rec, move=False):
    """Write ONE module's fit into <root>'s canonical layout (split from a
    multi-module file if needed). Returns the updated record. Idempotent:
    an existing fit_id is verified, not rewritten."""
    root = pathlib.Path(root)
    rp = root / rel_path(rec)
    dst = rp.with_suffix(".safetensors")
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = pathlib.Path(rec["location"].split("#", 1)[0])
    if not dst.exists():
        h, base = read_header(src)
        if len(modules_in(h)) == 1 and set(h) == {rec["module"] + s for s in _SUFFIXES}:
            shutil.copyfile(src, dst.with_suffix(".tmp"))
        else:
            _extract(src, h, base, rec["module"], dst.with_suffix(".tmp"))
        os.replace(dst.with_suffix(".tmp"), dst)
    h2, base2 = read_header(dst)
    again = describe(dst, h2, base2, rec["module"])
    if again["fit_id"] != rec["fit_id"]:
        raise RuntimeError(f"filed copy of {rec['module']} does not match its "
                           f"fit_id ({again['fit_id']} != {rec['fit_id']})")
    full = hashlib.sha256()
    with open(dst, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            full.update(blk)
    rec = {**rec, "location": str(dst), "sha256": full.hexdigest(),
           "filed": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    rp.with_suffix(".json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    return rec


def _extract(src, h, base, module, out):
    """Byte-exact slice of one module's three tensors into a new file."""
    keys = [module + s for s in _SUFFIXES]
    hdr, off, blobs = {}, 0, []
    with open(src, "rb") as f:
        for k in keys:
            a, b = h[k]["data_offsets"]
            f.seek(base + a)
            blobs.append(f.read(b - a))
            hdr[k] = {"dtype": h[k]["dtype"], "shape": h[k]["shape"],
                      "data_offsets": [off, off + (b - a)]}
            off += b - a
    hj = json.dumps(hdr, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(out, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for bl in blobs:
            f.write(bl)


def query(recs, family=None, teacher=None, layers=None, proj=None,
          d=None, K=None, recipe_known=None):
    out = []
    for r in recs.values():
        if family and r["family"] != family:
            continue
        if teacher and teacher not in r["teacher"]:
            continue
        if layers is not None and r.get("layer") not in layers:
            continue
        if proj and r["proj"] != proj:
            continue
        if d and r["d"] != d or K and r["K"] != K:
            continue
        if recipe_known is not None and bool(r.get("recipe")) != recipe_known:
            continue
        out.append(r)
    return sorted(out, key=lambda r: (r["family"], r["teacher"], r.get("layer") or -1,
                                      r["proj"], r["d"], r["K"]))
