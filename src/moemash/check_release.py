#!/usr/bin/env python3
"""Release gate: every file a USER needs exists and FUNCTIONS.

Third exhibit in two days of an artifact passing structural checks while
being unusable (vision missing -> check_vision; zero-byte packing ->
byte-aligned skip; and now a cheap-shallow 397B with NO TOKENIZER that
loaded "successfully" and encoded 16k chars to zero tokens). Presence is
not function: the tokenizer here must round-trip a non-trivial string.

    ./check_release.py --artifact <dir>
"""
import argparse
import json
import pathlib
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--artifact", required=True)
args = ap.parse_args()
A = pathlib.Path(args.artifact)

REQUIRED = ["config.json", "model.safetensors.index.json", "tokenizer.json",
            "tokenizer_config.json"]
fails = []
for f in REQUIRED:
    if not (A / f).exists():
        fails.append(f"MISSING {f}")
cfg = json.load(open(A / "config.json")) if (A / "config.json").exists() else {}
if cfg.get("model_file") and not (A / cfg["model_file"]).exists():
    fails.append(f"config names model_file={cfg['model_file']} but it is absent")

# index integrity: every mapped shard exists
if (A / "model.safetensors.index.json").exists():
    wm = json.load(open(A / "model.safetensors.index.json"))["weight_map"]
    for sh in sorted(set(wm.values())):
        if not (A / sh).exists():
            fails.append(f"index names missing shard {sh}")

# the tokenizer must FUNCTION, not merely exist (the failure that bit us
# loaded fine and encoded everything to zero tokens)
if (A / "tokenizer.json").exists():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(A))
        probe = "The harbourmaster recorded 417 brass lanterns at dawn."
        ids = tok.encode(probe)
        if len(ids) < 5:
            fails.append(f"tokenizer encodes probe to {len(ids)} tokens")
        elif probe not in tok.decode(ids):
            fails.append("tokenizer round-trip does not contain the input")
    except Exception as e:
        fails.append(f"tokenizer failed to load/encode: {e}")

if fails:
    print("FAIL:")
    for f in fails:
        print(f"    {f}")
    sys.exit(1)
print(f"PASS: {len(REQUIRED)} required files present, index complete, "
      f"tokenizer round-trips")
