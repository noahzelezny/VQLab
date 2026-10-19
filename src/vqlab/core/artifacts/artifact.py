"""The one place that knows how an artifact directory is laid out.

An artifact is: a config.json (with the three per-module maps vq_modules,
quantization and vq_skipzero), a model.safetensors.index.json (tensor ->
shard), shards, and other files (model.py, tokenizer, card). Every writer
used to re-derive "which tensors belong to which module, which shard holds
them, which config entries describe them" on its own, and every bug fixed on
2026-10-02 was one of those copies disagreeing with the others (a module
straddling a shard boundary, a skip-zero module refit, a group size).

This module answers those questions once. Writers read through it and write
through it; tests/test_writer_contracts.py pins what they produce.
"""
from __future__ import annotations

import collections
import json
import os
import pathlib
import re
import shutil
import struct
import subprocess
from functools import cached_property

INDEX = "model.safetensors.index.json"
SINGLE = "model.safetensors"
CONFIG = "config.json"
RECORDS = ("vqlab_provenance.json", "vqlab_provenance.history.jsonl")
MAPS = ("vq_modules", "quantization", "vq_skipzero")
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def module_of(key: str) -> str:
    """'a.b.gate_proj.codes' -> 'a.b.gate_proj'."""
    return key.rsplit(".", 1)[0]


def layer_of(key: str) -> int:
    """Layer number of a tensor or module name, -1 for top-level tensors."""
    m = _LAYER.search(key)
    return int(m.group(1)) if m else -1


def read_header(path, *, with_base: bool = False, metadata: bool = False):
    """A safetensors file's header: {tensor: {dtype, shape, data_offsets}}.
    The 8-byte little-endian length, then that many bytes of JSON. With
    with_base, returns (header, base): tensor bytes start at base +
    data_offsets[0]. `__metadata__` is dropped unless metadata=True."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    if not metadata:
        h.pop("__metadata__", None)
    return (h, 8 + n) if with_base else h


class Artifact:
    def __init__(self, path):
        self.dir = pathlib.Path(path)

    @classmethod
    def open(cls, path) -> "Artifact":
        a = cls(path)
        if not (a.dir / CONFIG).exists():
            raise SystemExit(f"{a.dir}: no {CONFIG}")
        return a

    # ---------------------------------------------------------------- reading
    @cached_property
    def config(self) -> dict:
        return json.loads((self.dir / CONFIG).read_text())

    @cached_property
    def index(self) -> dict:
        """tensor -> shard file name. A single-file checkpoint (a lone
        model.safetensors, common for small models) has no index file: its
        header is the map."""
        if not (self.dir / INDEX).exists() and (self.dir / SINGLE).exists():
            return dict.fromkeys(read_header(self.dir / SINGLE), SINGLE)
        return json.loads((self.dir / INDEX).read_text())["weight_map"]

    def map(self, name: str) -> dict:
        """One of vq_modules / quantization / vq_skipzero; module entries only
        (quantization's top-level group_size/bits/mode are not modules)."""
        return {k: v for k, v in (self.config.get(name) or {}).items() if isinstance(v, dict)}

    @property
    def vq_modules(self) -> set:
        return set(self.map("vq_modules"))

    @cached_property
    def shards(self) -> list:
        return sorted(set(self.index.values()))

    @cached_property
    def modules(self) -> dict:
        """module -> its tensor names."""
        out = collections.defaultdict(list)
        for k in self.index:
            out[module_of(k)].append(k)
        return dict(out)

    def shards_of(self, module: str) -> set:
        return {self.index[k] for k in self.modules.get(module, [])}

    def straddling(self) -> dict:
        """modules whose tensors sit in more than one shard -> those shards."""
        return {m: s for m in self.modules if len(s := self.shards_of(m)) > 1}

    def shard_layers(self, only: set | None = None) -> dict:
        """shard -> layers of the modules it holds (only those in `only`, when
        given; every shard is present, possibly with an empty set)."""
        out = {f: set() for f in self.shards}
        for k, f in self.index.items():
            if only is not None and module_of(k) not in only:
                continue
            out[f].add(layer_of(k))
        return out

    def header(self, shard: str) -> dict:
        return read_header(self.dir / shard)

    def shard_path(self, shard: str) -> str:
        return os.path.realpath(self.dir / shard)

    def other_files(self):
        """Non-shard, non-index, non-config, non-record files (model.py,
        tokenizer, card, template...)."""
        skip = {CONFIG, INDEX, *RECORDS}
        return [f for f in self.dir.iterdir() if f.is_file()
                and not f.name.endswith(".safetensors") and f.name not in skip]

    def size(self) -> int:
        return sum(os.path.getsize(self.shard_path(f)) for f in self.shards)


def merge_maps(sources) -> dict:
    """[(Artifact, modules)] -> {map: {module: entry}}: each module's entries
    come from the source whose bytes it takes; a module that is VQ anywhere
    loses any affine `quantization` entry."""
    merged = {m: {} for m in MAPS}
    for art, mods in sources:
        for name in MAPS:
            for mod, e in art.map(name).items():
                if mod in mods:
                    merged[name][mod] = e
    for mod in merged["vq_modules"]:
        merged["quantization"].pop(mod, None)
    return merged


def place(src, dst, copy=False):
    """Put a shard into an output dir: a symlink, or (copy) an APFS clone
    with a plain copy as fallback."""
    if copy:
        if subprocess.run(["cp", "-c", src, str(dst)], capture_output=True).returncode:
            shutil.copy(src, dst)
    else:
        os.symlink(src, dst)


def tensor_bytes(out, shards) -> int:
    """metadata.total_size as HF defines it: the sum of tensor DATA bytes
    across the shards (headers excluded), read from the shards' headers."""
    out = pathlib.Path(out)
    n = 0
    for f in set(shards):
        for v in read_header(out / f).values():
            a, b = v["data_offsets"]
            n += b - a
    return n


def write_index(out, weight_map, total_size="auto", metadata=None) -> None:
    """Write model.safetensors.index.json. total_size="auto" (the default)
    measures it from the output shards' headers, so it can never describe a
    parent's bytes; None omits it."""
    out = pathlib.Path(out)
    md = dict(metadata or {})
    if total_size == "auto":
        total_size = tensor_bytes(out, weight_map.values())
    if total_size is not None:
        md["total_size"] = total_size
    (out / INDEX).write_text(json.dumps({"metadata": md, "weight_map": weight_map}, indent=1))


def write_config(out, cfg) -> None:
    (pathlib.Path(out) / CONFIG).write_text(json.dumps(cfg, indent=1))


def copy_other_files(src: Artifact, out) -> None:
    for f in src.other_files():
        shutil.copy(f, out)


# ------------------------------------------------------------------- sizes
# Which tensors are NOT the text model. This is THE classifier: every writer,
# gate and bench that asks "is this the tower / the MTP head?" calls
# tensor_class (operator notes 2026-10-03, 1.5: three bugs in one night were
# name heuristics disagreeing). Spellings per family, measured on the shipped
# fleet (2026-10-02): Qwen3.5 model.visual, Qwen3.6 vision_tower, GLM
# vision_model, gemma vision_tower + embed_vision, DeepSeek-V4-Flash-Vision-Exp
# vision/aligner plus its image tokens (image_* in the release, model.image_*
# in the runtime) and the image-routing gate bias every layer carries
# (layers.N.ffn.gate.bias_vl). Any path SEGMENT naming vision/visual is a
# tower too, so a new spelling of the tower (vision_smoke's lesson: four
# spellings already) is not silently billed as text. An MTP head is either
# indexed (397B: top-level block.* + fc.*; DeepSeek source: mtp.*; a native
# runtime layout: language_model.mtp.*; DeepSeek-V3 style nextn.*) or a sidecar file
# (mtp-head*.safetensors) that the index does not list. GLM's MTP layer is
# spelled as an ordinary layer index and cannot be told apart by name.
TOWER_PREFIXES = ("model.visual.", "visual.", "vision_tower.", "vision_model.",
                  "vision.", "aligner.", "model.vision_tower.", "model.vision_model.",
                  "model.aligner.", "embed_vision.", "model.embed_vision.",
                  "image_", "model.image_")
TOWER_SUFFIXES = (".bias_vl",)
MTP_PREFIXES = ("block.", "fc.", "mtp.", "model.mtp.")
MTP_SEGMENTS = ("mtp", "nextn")


# The teacher's tokenizer files, copied byte for byte over what
# load_tokenizer(...).save_pretrained writes (stream-convert). A re-serialization is not a copy:
# transformers 5 rewrote the Qwen3.5 tokenizer.json's ByteLevel pre_tokenizer
# and decoder (add_prefix_space/trim_offsets/use_regex flipped), and the
# 2026-10-13 397B preflight's check-release failed the artifact against its own
# teacher on exactly that; the published builds had needed a hand fix
# (provenance --amend tokenizer.json, 2026-10-10).
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "added_tokens.json", "vocab.json", "merges.txt", "tokenizer.model",
                   "chat_template.jinja", "chat_template.json")


def copy_tokenizer_files(src, dst) -> list[str]:
    """Copy every TOKENIZER_FILES entry `src` has into `dst`, verbatim
    (replacing a re-serialized one) -> the names copied."""
    src, dst = pathlib.Path(src), pathlib.Path(dst)
    got = []
    for name in TOKENIZER_FILES:
        if (src / name).is_file():
            shutil.copyfile(src / name, dst / name)
            got.append(name)
    return got


# The teacher's generation and processor configs. A vision model whose build
# lacks its processor config cannot serve an image request (check-release
# refuses it): GLM's struct base once copied generation_config.json alone.
# One of IMAGE_PROCESSOR_FILES is what an image request needs (check-release).
IMAGE_PROCESSOR_FILES = ("preprocessor_config.json", "processor_config.json")
PROCESSOR_FILES = (*IMAGE_PROCESSOR_FILES, "video_preprocessor_config.json")
CONFIG_FILES = ("generation_config.json", *PROCESSOR_FILES)


def copy_config_files(src, dst, names: tuple[str, ...] = CONFIG_FILES,
                      overwrite: bool = True) -> list[str]:
    """Copy each of `names` that `src` has into `dst` (keeping one `dst`
    already has unless `overwrite`) -> the names copied."""
    src, dst = pathlib.Path(src), pathlib.Path(dst)
    got = [n for n in names
           if (src / n).is_file() and (overwrite or not (dst / n).exists())]
    for name in got:
        shutil.copyfile(src / name, dst / name)
    return got


def tensor_class(key: str) -> str:
    """'text', 'tower' (vision tower, projector, image tokens, image-routing
    biases) or 'mtp' (an indexed draft head)."""
    segs = key.split(".")
    if key.startswith(TOWER_PREFIXES) or key.endswith(TOWER_SUFFIXES) \
            or any("vision" in s or "visual" in s for s in segs):
        return "tower"
    if key.startswith(MTP_PREFIXES) or any(s.lower() in MTP_SEGMENTS for s in segs):
        return "mtp"
    return "text"


def sizes(path) -> dict:
    """Bytes three ways (memory note artifact-size-three-ways): text weights,
    the vision tower, an MTP head (indexed or sidecar), and the full download
    (every file in the dir). Quote TEXT as the headline or sizes never
    reconcile across cards."""
    a = Artifact.open(path)
    out = {"text": 0, "tower": 0, "mtp": 0}
    for f in a.shards:
        for k, v in a.header(f).items():
            s, e = v["data_offsets"]
            out[tensor_class(k)] += e - s
    listed = set(a.shards)
    side = [p for p in a.dir.glob("mtp-head*.safetensors") if p.name not in listed]
    out["mtp_sidecar"] = sum(p.stat().st_size for p in side)
    out["mtp_files"] = sorted(p.name for p in side)
    out["download"] = sum(p.stat().st_size for p in a.dir.iterdir()
                          if p.is_file() and not p.name.startswith("."))
    return out
