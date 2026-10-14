"""vqlab lab scan — find the models already on this machine.

    vqlab lab scan [--json] [--add DIR]...

Looks where Mac model tools keep their downloads and lists every model it
finds, so the GUI can show them beside vqlab's own without copying or
linking anything. It never writes outside the index.

Places, each a standard default (moved ones follow their tool's env var):

  vqlab models / teachers   config.models(), config.teachers()
  Hugging Face cache        $HF_HUB_CACHE, $HF_HOME/hub, ~/.cache/huggingface/hub
                            (mlx-lm, transformers, `hf download`)
  LM Studio                 ~/.lmstudio/models, ~/.cache/lm-studio/models
  exo                       $EXO_HOME/models, ~/.exo/models
  Ollama                    ~/.ollama/models (GGUF blobs: counted, not readable)
  your own folders          [scan] extra = [...] in the config file, or --add

Only each model's config.json and file list are read. Each model is tagged:

  original     the released precision (bf16 / fp16 / fp32, or a release that
               ships fp8 itself): a recipe can build from it
  quantized    quantized weights (mlx affine, vqlab VQ, ...): run or compare
  incomplete   a download with no weights in it yet
  unreadable   weights vqlab cannot read (GGUF and the like)

`family` names the vqlab family that handles the model_type, or is null
when vqlab cannot build from that architecture yet.

A model reachable through two places (a symlinked folder, a shared cache)
is listed once, under the first place that found it. The index is
<records>/registry/found.jsonl, rewritten on every scan.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import sys
import threading

from vqlab import config
from vqlab.util.fs import GIB, read_jsonl, write_text_atomic

FULL = {"bfloat16", "float16", "float32", "bf16", "fp16", "fp32"}


def index_path() -> pathlib.Path:
    return config.registry_dir() / "found.jsonl"


def _home() -> pathlib.Path:
    return pathlib.Path.home()


def places(extra=()) -> list[tuple[str, pathlib.Path, str]]:
    """(source label, folder, layout) for every place worth looking.
    layout: 'flat' (one folder per model, maybe nested one level as
    org/name), 'hf' (models--org--name/snapshots/<rev>), 'ollama'."""
    h = _home()
    exo = pathlib.Path(os.environ["EXO_HOME"]) / "models" if os.environ.get("EXO_HOME") \
        else h / ".exo" / "models"
    out = [("vqlab models", config.models(), "flat"),
           ("vqlab teachers", config.teachers(), "flat"),
           ("Hugging Face cache", config.hf_cache(), "hf"),
           ("LM Studio", h / ".lmstudio" / "models", "flat"),
           ("LM Studio", h / ".cache" / "lm-studio" / "models", "flat"),
           ("exo", exo, "flat"),
           ("Ollama", h / ".ollama" / "models", "ollama")]
    cfg = config._file_table(str(config.config_file()), "scan").get("extra", [])
    for d in [*cfg, *extra]:
        out.append(("your folder", pathlib.Path(d).expanduser(), "flat"))
    return out


def _listdir(d: pathlib.Path) -> list[pathlib.Path]:
    """d's entries, or [] when d cannot be read: one unreadable folder costs
    that folder, never the rest of the place."""
    try:
        return sorted(d.iterdir())
    except OSError:
        return []


def _mtime(p: pathlib.Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _is_model(d: pathlib.Path) -> bool:
    return any(f.name == "config.json" or f.suffix == ".gguf" for f in _listdir(d))


def _model_dirs(root: pathlib.Path, layout: str):
    """(display name, model folder) under one place."""
    if layout == "hf":
        for repo in (p for p in _listdir(root) if p.name.startswith("models--")):
            snaps = [s for s in _listdir(repo / "snapshots") if s.is_dir()]
            if snaps:
                latest = max(snaps, key=_mtime)
                yield repo.name[len("models--"):].replace("--", "/", 1), latest
        return
    for d in (p for p in _listdir(root) if p.is_dir() and not p.name.startswith(".")):
        if _is_model(d):
            yield d.name.replace("--", "/", 1), d
        else:                                       # publisher/repo (LM Studio)
            for sub in (p for p in _listdir(d) if p.is_dir()):
                if _is_model(sub):
                    yield f"{d.name}/{sub.name}", sub


def _size(d: pathlib.Path) -> int:
    n = 0
    for f in d.iterdir():
        try:
            if f.is_file():
                n += f.stat().st_size              # follows the HF cache's blob symlinks
        except OSError:
            pass
    return n


def classify(d: pathlib.Path) -> dict:
    """Read config.json and the file list only."""
    names = [f.name for f in d.iterdir()]
    st = [n for n in names if n.endswith(".safetensors")]
    if not st:
        if any(n.endswith(".gguf") for n in names):
            return {"use": "unreadable", "format": "GGUF"}
        return {"use": "incomplete", "format": "no weights downloaded"}
    try:
        cfg = json.loads((d / "config.json").read_text())
    except (OSError, ValueError):
        return {"use": "unreadable", "format": "safetensors without a readable config.json"}
    if not isinstance(cfg, dict):
        return {"use": "unreadable", "format": "config.json is not an object"}
    text = cfg.get("text_config") or {}
    q = cfg.get("quantization") or cfg.get("quantization_config") or {}
    mt = cfg.get("model_type")
    from vqlab import family
    fam = family.get(mt) if mt else None
    out = {"model_type": mt, "family": fam.name if fam else None, "shards": len(st)}
    if cfg.get("vq_modules"):
        out.update(use="quantized", format="vqlab VQ")
    elif str(q.get("quant_method", "")).lower() == "fp8" and not q.get("bits"):
        out.update(use="original", format="fp8 (as released)")
    elif q:
        bits = q.get("bits")
        out.update(use="quantized", format=(f"{bits}-bit " if bits else "")
                   + str(q.get("quant_method") or q.get("mode") or "mlx affine"))
        if bits:
            out["bits"] = bits
    else:
        dt = str(cfg.get("torch_dtype") or cfg.get("dtype") or text.get("torch_dtype")
                 or text.get("dtype") or "").lower()
        out.update(use="original" if dt in FULL or not dt else "quantized",
                   format=dt or "unstated dtype")
    return out


def _ollama(root: pathlib.Path) -> list[dict]:
    man = root / "manifests"
    if not man.is_dir():
        return []
    tags = [p for p in man.rglob("*") if p.is_file()]
    if not tags:
        return []
    blobs = root / "blobs"
    size = sum(f.stat().st_size for f in blobs.iterdir() if f.is_file()) if blobs.is_dir() else 0
    return [{"source": "Ollama", "name": f"{len(tags)} Ollama model(s)", "path": str(root),
             "use": "unreadable", "format": "Ollama GGUF blobs", "bytes": size,
             "count": len(tags)}]


def scan(extra=()) -> list[dict]:
    """Every model found, one row each; never raises on an unreadable place."""
    rows, seen = [], set()
    for source, root, layout in places(extra):
        try:
            if not root.is_dir():
                continue
            if layout == "ollama":
                rows += _ollama(root)
                continue
            for name, d in _model_dirs(root, layout):
                real = os.path.realpath(d)
                if real in seen:
                    continue
                seen.add(real)
                try:
                    row = {"source": source, "name": name, "path": str(d),
                           **classify(d), "bytes": _size(d)}
                except Exception as e:  # classification is best effort: one odd model, one row
                    row = {"source": source, "name": name, "path": str(d),
                           "use": "unreadable", "format": f"unreadable ({type(e).__name__})"}
                rows.append(row)
        except OSError:
            continue
    return rows


def write(rows) -> pathlib.Path:
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    return write_text_atomic(index_path(), "".join(json.dumps({**r, "scanned": stamp}) + "\n"
                                                   for r in rows))


def load() -> list[dict]:
    """The last scan's rows ([] before the first); a torn line costs that row."""
    return read_jsonl(index_path())


def refresh(extra=()) -> list[dict]:
    rows = scan(extra)
    write(rows)
    return rows


_SCAN_LOCK = threading.Lock()      # held while a background scan runs


def scanning() -> bool:
    return _SCAN_LOCK.locked()


def refresh_in_background() -> bool:
    """Rescan on a daemon thread, without holding up the caller (the GUI at
    launch, Rescan, a finished download). A scan already running is not
    started twice; returns whether this call started one."""
    if not _SCAN_LOCK.acquire(blocking=False):
        return False

    def go():
        try:
            refresh()
        except Exception as e:  # report, never kill the server that asked
            print(f"vqlab lab scan: background scan failed: {type(e).__name__}: {e}", file=sys.stderr)
        finally:
            _SCAN_LOCK.release()
    threading.Thread(target=go, daemon=True).start()
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vqlab lab scan", description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--add", action="append", default=[], metavar="DIR",
                    help="also look in DIR (this scan only; keep it with [scan] extra)")
    a = ap.parse_args(argv)
    rows = refresh(a.add)
    if a.json:
        print(json.dumps(rows, indent=1))
        return 0
    for r in rows:
        gib = (r.get("bytes") or 0) / GIB
        print(f"  {r['use']:10s} {r['source']:18s} {r['name'][:52]:52s} "
              f"{gib:8.1f} GiB  {r.get('format', '')}")
    by = {}
    for r in rows:
        by[r["use"]] = by.get(r["use"], 0) + r.get("count", 1)
    print(f"{len(rows)} entries ({', '.join(f'{n} {k}' for k, n in sorted(by.items()))}) "
          f"-> {index_path()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
