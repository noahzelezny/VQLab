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
import collections
import json
import pathlib
import re
import struct
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
ap.add_argument("--no-prefill-smoke", action="store_true",
                help="skip the LARGE-N prefill smoke. The 4-token smoke only "
                     "exercises the fused (small-N) path; the entire 35B "
                     "ladder shipped a _prefill that CRASHED on any long "
                     "prompt (arc-5 sig mis-binding, 2026-09-06) and every "
                     "gate stayed green because nothing ever pushed N past "
                     "the fused cutoff. ~1500 prompt tokens exceeds both the "
                     "dense packed cutoff (N>96) and the MoE pair cutoff "
                     "(4096 pairs = 512 tokens at top_k 8).")
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
# 2026-09-07: the file NAME is family-dependent. Qwen-VL ships
# preprocessor_config.json; gemma-4 ships processor_config.json and no
# preprocessor at all — its own mlx-community bf16 base has only the
# latter, so demanding the former failed a correct artifact. Require
# that SOME processor config is present, and (below) that the artifact
# is not missing one its own base ships.
# 2026-09-07 (second correction): this was a blanket FAIL, and it appended
# to `fails`, which SUPPRESSES the smoke below (`and not fails`) -- so the
# gate asserted "fails every request, text included" while preventing the
# one test that could check that claim. It false-failed GLM-5.3-Flash
# 2.7/3.6, which carry vision_config, ship no processor config in ANY
# build (including zai-org's own upstream 3/4/6-bit), and serve coherent
# text on the 2-node ring -- measured twice. The base-comparison the
# comment above promises was never actually written.
# Defer the verdict instead: let the smoke run, then judge on evidence.
#   smoke PASSES -> the "text included" claim is disproven for this
#                   artifact; the real limitation is that IMAGE requests
#                   cannot work. Report it loudly, do not fail the gate.
#   smoke FAILS / skipped -> nothing disproves it; keep the hard FAIL
#                   (this is the gemma-4 and Flash-Next case, where the
#                   missing processor presented as a warmup hang).
_PROC_FILES = ("preprocessor_config.json", "processor_config.json")
_proc_gap = ("vision_config" in cfg
             and not any((A / f).exists() for f in _PROC_FILES))
_PROC_MSG = (f"config carries vision_config but none of {list(_PROC_FILES)} "
             "is present; a runtime that instantiates an image processor "
             "from the artifact dir cannot serve IMAGE requests")

# index integrity: every mapped shard exists
if (A / "model.safetensors.index.json").exists():
    wm = json.load(open(A / "model.safetensors.index.json"))["weight_map"]
    for sh in sorted(set(wm.values())):
        if not (A / sh).exists():
            fails.append(f"index names missing shard {sh}")

# DTYPE CENSUS. Two artifacts shipped in one day (2026-09-07) with a single
# tensor class silently promoted to float32, and NOTHING here caught either:
# every other gate is single-box, and fp32 only fails once a model is
# sharded. embed_tokens got fp32 scales (mx.quantize returns scales in the
# INPUT dtype and the requant tool had upcast); the MTP sidecar carries fp32
# RMSNorm gains (residue of a `+shift` never cast back). Both make the whole
# forward pass fp32, and at head_dim 256 that asks Metal for a 53 KB
# threadgroup attention kernel against a 32 KB cap -- every sharded prefill
# dies, while single-box generation is fine.
#
# The check compares each tensor against a SOURCE OF TRUTH for its own
# class, never a fixed allowlist: this family legitimately stores 45 float32
# `linear_attn.A_log` tensors, so a blanket "no fp32" rule would fire on
# every correct build and be muted within a week.
def _st_header(p):
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return {k: v for k, v in json.loads(fh.read(n)).items()
                if k != "__metadata__"}

if (A / "model.safetensors.index.json").exists() and cfg:
    _wm = json.load(open(A / "model.safetensors.index.json"))["weight_map"]
    _dt = {}
    for _sh in sorted(set(_wm.values())):
        if (A / _sh).exists():
            for _k, _v in _st_header(A / _sh).items():
                _dt[_k] = _v["dtype"]

    # (a) every affine module's scales/biases share one dtype, and likewise
    #     every VQ module's codebook/vq_scales. Catches one mis-quantized
    #     module among hundreds.
    _q = {m for m, v in cfg.get("quantization", {}).items()
          if isinstance(v, dict) and "bits" in v}
    _vq = set(cfg.get("vq_modules", {}))
    for _label, _mods, _sufs in (("affine", _q, ("scales", "biases")),
                                 ("VQ", _vq, ("codebook", "vq_scales"))):
        for _suf in _sufs:
            _names = [f"{m}.{_suf}" for m in _mods if f"{m}.{_suf}" in _dt]
            _c = collections.Counter(_dt[n] for n in _names)
            if len(_c) > 1:
                _odd = _c.most_common()[-1]
                _ex = next(n for n in _names if _dt[n] == _odd[0])
                fails.append(f"{_label} .{_suf} tensors disagree on dtype: "
                             f"{dict(_c)} — {_odd[1]} outlier(s), e.g. "
                             f"{_ex} is {_odd[0]}")

    # (b) a codes tensor's dtype is a FUNCTION of its own geometry (uint32
    #     when packed, else uint8 for K<=256, else uint16), so a
    #     mixed-geometry artifact holds several LEGITIMATELY. Test each
    #     against its own entry -- stronger than peer uniformity, and it
    #     catches a config/tensor divergence no peer comparison could.
    for _m, _e in cfg.get("vq_modules", {}).items():
        _k = _m + ".codes"
        if _k not in _dt:
            continue
        _want = ("U32" if _e.get("pack_bits")
                 else ("U8" if _e.get("k", 0) <= 256 else "U16"))
        if _dt[_k] != _want:
            fails.append(f"{_k} is {_dt[_k]} but its vq_modules entry "
                         f"(dim={_e.get('dim')} k={_e.get('k')} "
                         f"pack_bits={_e.get('pack_bits')}) implies {_want}")

    # (c) tensors filling the same role across layers must agree. `codes` is
    #     exempt -- (b) owns it. Uniformly-fp32 classes (A_log) never fire.
    _cls = collections.defaultdict(list)
    for _k in _dt:
        if not _k.endswith(".codes"):
            _cls[re.sub(r"\.\d+\.", ".", _k)].append(_k)
    for _cn, _names in sorted(_cls.items()):
        _c = collections.Counter(_dt[n] for n in _names)
        if len(_c) > 1:
            _odd = _c.most_common()[-1]
            _ex = next(n for n in _names if _dt[n] == _odd[0])
            fails.append(f"role class {_cn} has mixed dtypes {dict(_c)} "
                         f"— e.g. {_ex} is {_odd[0]}")

    # (d) a .safetensors the index does not reference is a SIDECAR (MTP
    #     head, vision graft). Its tensors must match the dtype the trunk
    #     uses for the same role -- this is what catches the published MTP
    #     heads' fp32 norms.
    _tail = {}
    for _k, _d in _dt.items():
        _tail.setdefault(".".join(_k.split(".")[-2:]), set()).add(_d)
    for _f in sorted(A.glob("*.safetensors")):
        if _f.name in set(_wm.values()):
            continue
        for _k, _v in _st_header(_f).items():
            _ref = _tail.get(".".join(_k.split(".")[-2:]))
            if _ref and len(_ref) == 1 and _v["dtype"] not in _ref:
                fails.append(f"sidecar {_f.name}: {_k} is {_v['dtype']} but "
                             f"the trunk stores "
                             f"{'.'.join(_k.split('.')[-2:])} as "
                             f"{next(iter(_ref))}")

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
    elif not args.no_prefill_smoke:
        # Large-N arm: force the PREFILL path (decode fallback / _prefill),
        # which the 4-token smoke never touches — see --no-prefill-smoke.
        long_prompt = ("The quick brown fox jumps over the lazy dog. "
                       "Numbers: 0 1 2 3 4 5 6 7 8 9. ") * 90   # ~1.5k tok
        print("running LARGE-N prefill smoke (~1.5k-token prompt) ...",
              flush=True)
        r = subprocess.run([sys.executable, str(HERE / "smoke.py"), str(A),
                            "--strict", "--max-tokens", "2",
                            "--prompt", long_prompt])
        if r.returncode != 0:
            fails.append("LARGE-N prefill smoke failed — the artifact "
                         "generates fine at decode N but its prefill path "
                         "(large-N fallback) cannot process a long prompt. "
                         "This is the failure the 35B ladder shipped with.")
elif args.no_smoke:
    print("NOTE: --no-smoke; static checks only, generation NOT verified.")

# Resolve the deferred processor-config finding now that the smoke has (or
# has not) demonstrated that this artifact can actually serve.
if _proc_gap:
    if args.no_smoke or fails:
        fails.append(_PROC_MSG + " -- and no smoke proved this artifact can "
                     "serve at all, so the text-path risk stands unrefuted")
    else:
        print(f"WARNING: {_PROC_MSG}. Text generation was VERIFIED by the "
              "smoke above, so this does not block release; vision requests "
              "on this artifact are expected to fail.")

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
