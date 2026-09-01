#!/usr/bin/env python3
"""Release gate: every file a USER needs exists and FUNCTIONS.

Third exhibit in two days of an artifact passing structural checks while
being unusable (vision missing -> check_vision; zero-byte packing ->
byte-aligned skip; and now a cheap-shallow 397B with NO TOKENIZER that
loaded "successfully" and encoded 16k chars to zero tokens). Presence is
not function: the tokenizer here must round-trip a non-trivial string.

Fourth exhibit, 2026-09-01: three dense 27B rungs shipped a bundled model.py
importing `mlx_lm.models.vq_switch`, a module present only in our development
venvs. Two could not generate a token for anyone who downloaded them. Every
structural check passed, because the bytes were all there -- the defect was
in what the bundle REACHED FOR at runtime.

So this gate now does two more things: a static scan of the bundle for
imports a downloader cannot satisfy (cheap, no model load), and a real
`smoke --strict` generation that asserts the runtime resolved from the
artifact itself. Nothing should be uploaded that has not passed this.

    ./check_release.py --artifact <dir> [--no-smoke]
"""
import argparse
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--artifact", required=True)
ap.add_argument("--no-smoke", action="store_true",
                help="skip the strict generation smoke. The static checks "
                     "still run, but nothing here then proves the artifact "
                     "can produce a token on a machine that is not ours -- "
                     "which is exactly how three broken rungs shipped.")
ap.add_argument("--max-tokens", type=int, default=4)
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

# The bundled runtime must not reach for anything a downloader lacks. This
# is a byte check -- no model load, no GPU -- and it alone would have caught
# all three broken rungs the moment they were built.
_mf = cfg.get("model_file")
if _mf and (A / _mf).exists():
    _src = (A / _mf).read_text()
    for _bad in ("from mlx_lm.models.vq_", "import mlx_lm.models.vq_"):
        if _bad in _src:
            _lines = [i + 1 for i, l in enumerate(_src.splitlines())
                      if _bad in l]
            fails.append(
                f"{_mf} contains {_bad!r} at line(s) "
                f"{_lines[:6]} — no released mlx-lm has those modules, so "
                f"this bundle cannot run outside our venvs")
    try:
        compile(_src, _mf, "exec")
    except SyntaxError as e:
        fails.append(f"{_mf} does not compile: {e}")

# A real token, through the artifact's own runtime, with the resolution
# assertions on. This is the step that costs a model load, and the one that
# actually certifies the thing.
if not args.no_smoke and not fails:
    print(f"running strict smoke ({args.max_tokens} token) ...", flush=True)
    r = subprocess.run([sys.executable, str(HERE / "smoke.py"), str(A),
                        "--strict", "--max-tokens", str(args.max_tokens)])
    if r.returncode != 0:
        fails.append("strict smoke failed — see its output above. The "
                     "artifact either could not generate, or resolved its "
                     "runtime from a copy a downloader does not have.")
elif args.no_smoke:
    print("NOTE: --no-smoke; static checks only, generation NOT verified.")

if fails:
    print("FAIL:")
    for f in fails:
        print(f"    {f}")
    sys.exit(1)
_smoked = "" if args.no_smoke else ", strict smoke generated a token"
print(f"PASS: {len(REQUIRED)} required files present, index complete, "
      f"tokenizer round-trips, bundle imports nothing a downloader "
      f"lacks{_smoked}")
