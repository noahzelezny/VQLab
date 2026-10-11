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

from vqlab import config

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
    hf = os.environ.get("HF_HUB_CACHE") or (
        str(pathlib.Path(os.environ["HF_HOME"]) / "hub") if os.environ.get("HF_HOME")
        else str(h / ".cache" / "huggingface" / "hub"))
    exo = pathlib.Path(os.environ["EXO_HOME"]) / "models" if os.environ.get("EXO_HOME") \
        else h / ".exo" / "models"
    out = [("vqlab models", config.models(), "flat"),
           ("vqlab teachers", config.teachers(), "flat"),
           ("Hugging Face cache", pathlib.Path(hf).expanduser(), "hf"),
           ("LM Studio", h / ".lmstudio" / "models", "flat"),
           ("LM Studio", h / ".cache" / "lm-studio" / "models", "flat"),
           ("exo", exo, "flat"),
           ("Ollama", h / ".ollama" / "models", "ollama")]
    cfg = config._file_table(str(config.config_file()), "scan").get("extra", [])
    for d in [*cfg, *extra]:
        out.append(("your folder", pathlib.Path(d).expanduser(), "flat"))
    return out


def _model_dirs(root: pathlib.Path, layout: str):
    """(display name, model folder) under one place."""
    if layout == "hf":
        for repo in sorted(root.glob("models--*")):
            snaps = sorted((repo / "snapshots").glob("*"), key=lambda p: p.stat().st_mtime)
            if snaps:
                yield repo.name[len("models--"):].replace("--", "/", 1), snaps[-1]
        return
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if (d / "config.json").exists() or any(d.glob("*.gguf")):
            yield d.name.replace("--", "/", 1), d
        else:                                       # publisher/repo (LM Studio)
            for sub in sorted(p for p in d.iterdir() if p.is_dir()):
                if (sub / "config.json").exists() or any(sub.glob("*.gguf")):
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
                except OSError as e:
                    row = {"source": source, "name": name, "path": str(d),
                           "use": "unreadable", "format": f"unreadable ({e.strerror})"}
                rows.append(row)
        except OSError:
            continue
    return rows


def write(rows) -> pathlib.Path:
    p = index_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    tmp = p.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps({**r, "scanned": stamp}) + "\n" for r in rows))
    tmp.replace(p)
    return p


def load() -> list[dict]:
    p = index_path()
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def refresh(extra=()) -> list[dict]:
    rows = scan(extra)
    write(rows)
    return rows


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
        gib = (r.get("bytes") or 0) / 2**30
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
