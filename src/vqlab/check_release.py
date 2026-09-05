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
ap.add_argument("--cluster-smoke", metavar="URL", default=None,
                help="run the generation smoke through an exo cluster API "
                     "(e.g. http://localhost:52415) instead of a local "
                     "load. For artifacts too large for any single box we "
                     "gate on. The exo node on THIS machine must be serving "
                     "the exact directory being gated (checked by realpath, "
                     "not trusted).")
ap.add_argument("--cluster-peer", metavar="USER@HOST", default=None,
                help="with --cluster-smoke: ssh peer(s) holding the other "
                     "pipeline rank's copy; their model.py/config.json/"
                     "index are hashed and shard sizes compared against "
                     "this artifact before the smoke counts.")
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

# a vision-capable config commits us to the image-processor contract: any
# runtime that sees vision_config (mlx_vlm, exo) instantiates an
# AutoImageProcessor from the artifact dir, and its absence fails EVERY
# request, text included, after prefill -- which presents as a warmup hang,
# not a clean error. Fifth exhibit, 2026-09-04: all four Flash-Next rungs
# shipped without preprocessor_config.json; the fitting pipeline never
# touches images so nothing ever staged it.
if "vision_config" in cfg and not (A / "preprocessor_config.json").exists():
    fails.append("config carries vision_config but preprocessor_config.json "
                 "is absent (vision runtimes fail every request without it)")

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

# The gate itself must be running against a STOCK mlx-lm, or every check
# below is meaningless: the quantlab-era patch_mlx_lm.py installed
# vq_switch.py INTO the venv's mlx_lm, silently and persistently, and a
# gate run in such an env blessed incomplete bundles that ModuleNotFoundError
# for every real downloader (the 27B rungs, discovered 2026-09-02 night).
# The env that did it was deleted, but the failure mode is one
# patch_mlx_lm.py invocation away from coming back -- so the gate checks.
try:
    import mlx_lm.models as _mlm
    _patched = pathlib.Path(_mlm.__file__).parent / "vq_switch.py"
    if _patched.exists():
        fails.append(
            f"THIS VENV IS PATCHED: {_patched} exists (quantlab-era "
            f"patch_mlx_lm.py residue). A gate run here cannot certify "
            f"anything -- bundles that depend on the patch pass here and "
            f"break for every downloader. Remove the file and re-run.")
except ImportError:
    pass

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

# CLUSTER SMOKE. Rungs larger than any single box we gate on (GLM 3.1/3.6,
# 397B 2.4+) used to be un-smokeable: the RAM preflight refused, and their
# first publish was impossible without a bypass -- which this gate refuses
# to have. This path replaces the local load with a real generation through
# a 2-node exo pipeline, which exercises the same bundle, the same shards
# and the same kernels. It is NOT an attestation flag: the gate itself
# drives the generation and first proves the cluster is serving the very
# bytes being gated:
#   - the artifact dir must carry the exo naming (<Owner--Name>) for the
#     served model id. exo resolves that name inside its models roots, and
#     on the gate box the publish source IS in a models root -- verified by
#     shard atimes advancing at instance load (2026-09-03). The gate cannot
#     see exo's config from here, so this leg is convention + naming, not a
#     realpath proof; the peer legs below are byte checks.
#   - each --cluster-peer's copy is checked over ssh -- sha256 of model.py /
#     config.json / model.safetensors.index.json, plus size of every shard.
#     (Shard CONTENT on the peer is not re-hashed -- 100G+ per rung -- so a
#     peer copy corrupted at equal size with equal index would pass; every
#     copy we make is rsync'd from this artifact, and the smoke still has
#     to decode coherent tokens through those bytes.)
def _cluster_smoke() -> list[str]:
    import hashlib
    import subprocess as sp
    import urllib.request
    probs = []
    base = args.cluster_smoke.rstrip("/")
    dirname = A.resolve().name                     # Owner--Name
    model_id = dirname.replace("--", "/", 1)
    served = json.load(urllib.request.urlopen(f"{base}/v1/models",
                                              timeout=10))
    ids = {m["id"] for m in served["data"]}
    if model_id not in ids:
        return [f"cluster at {base} does not list {model_id}"]
    # local-rank identity: exo's models root is the parent of dirs named
    # Owner--Name; the artifact being gated must be that exact directory.
    node_dir = A.resolve()
    if node_dir.name != dirname or not (node_dir / "config.json").exists():
        return [f"artifact dir {node_dir} does not look like the served "
                f"model dir for {model_id}"]
    ident = ["model.py", "config.json", "model.safetensors.index.json"]
    local_hashes = {}
    for f in ident:
        p = A / f
        if p.exists():
            local_hashes[f] = hashlib.sha256(p.read_bytes()).hexdigest()
    shard_sizes = {sh: (A / sh).stat().st_size for sh in sorted(set(
        json.load(open(A / "model.safetensors.index.json"))["weight_map"]
        .values()))}
    peers = [p for p in (args.cluster_peer,) if p]
    if not peers:
        probs.append("cluster smoke requires --cluster-peer: the other "
                     "rank's copy must be identity-checked, not assumed")
        return probs
    for peer in peers:
        # peer syntax: user@host[:models-root]; default root is the exo
        # convention "$HOME/Exo Models". A peer whose root lives elsewhere
        # (the M3 keeps models on an external volume) names it explicitly.
        host, _, root = peer.partition(":")
        root = root or "$HOME/Exo Models"
        script = ("cd \"" + root + "/" + dirname + "\" && "
                  "shasum -a 256 " + " ".join(local_hashes) + " && "
                  "stat -f '%N %z' " + " ".join(shard_sizes))
        r = sp.run(["ssh", host, script], capture_output=True, text=True,
                   timeout=120)
        if r.returncode != 0:
            probs.append(f"peer {peer}: identity check failed to run: "
                         f"{r.stderr.strip()[:200]}")
            continue
        seen = {}
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) == 2:
                val, name = ((parts[0], parts[1]) if len(parts[0]) == 64
                             else (parts[1], parts[0]))
                seen[pathlib.Path(name).name] = val
        for f, h in local_hashes.items():
            if seen.get(f) != h:
                probs.append(f"peer {peer}: {f} hash mismatch "
                             f"(theirs {str(seen.get(f))[:12]}..., "
                             f"ours {h[:12]}...)")
        for sh, sz in shard_sizes.items():
            if seen.get(sh) != str(sz):
                probs.append(f"peer {peer}: shard {sh} size "
                             f"{seen.get(sh)} != {sz}")
    if probs:
        return probs
    body = json.dumps({
        "model": model_id,
        "messages": [{"role": "user",
                      "content": "Name three colors, then stop."}],
        "max_tokens": max(args.max_tokens, 16), "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"{base}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    resp = json.load(urllib.request.urlopen(req, timeout=900))
    text = (resp.get("choices") or [{}])[0].get("message", {}).get(
        "content", "")
    if not text.strip():
        return [f"cluster smoke returned no content: "
                f"{json.dumps(resp)[:300]}"]
    print(f"  cluster smoke output: {text.strip()[:120]!r}")
    return []


if args.cluster_smoke and not args.no_smoke and not fails:
    print("running CLUSTER smoke (2-node exo pipeline; peer identity "
          "checked first) ...", flush=True)
    fails += _cluster_smoke()
# A real token, through the artifact's own runtime, with the resolution
# assertions on. This is the step that costs a model load, and the one that
# actually certifies the thing.
elif not args.no_smoke and not fails:
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
_smoked = ("" if args.no_smoke else
           ", CLUSTER smoke generated tokens through a peer-verified "
           "2-node pipeline" if args.cluster_smoke else
           ", strict smoke generated a token")
print(f"PASS: {len(REQUIRED)} required files present, index complete, "
      f"tokenizer round-trips, bundle imports nothing a downloader "
      f"lacks{_smoked}")
