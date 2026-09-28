#!/usr/bin/env python3
"""vqlab reselect -- activation-aware (G-aware) code re-selection, as a tool.

    vqlab reselect calibrate --model ART --out BANK [--tokens 8192] [--dry-run]
    vqlab reselect apply --artifact ART --grams BANK --target TEACHER --out NEW
    vqlab reselect selftest [--dir D]      CPU-only synthetic check, no model

WHY THIS IS BEING RE-TESTED. F67-F75 measured that re-picking each
subvector's code (codebook FIXED) to minimize output-space error under a
block-diagonal activation Gram improves KL-to-teacher (mean -2.2% 35B /
-1.9% Flash, P99.9 tails -8..-10%), top-1 agreement and the 35B's task
benchmarks. F76-F78 then REJECTED it because wikitext ppl worsened (+0.45%
35B, +1.9% Flash-2.1) -- and F78 was judged by the ppl gate, which was the
release gate at the time. Since F118 (Noah, 2026-09-16) the release gate is
paired 3-corpus KL (`vqlab kl-ladder`, |t|>2); ppl is printed, not gated. A
method rejected on an instrument that is no longer the gate gets re-tested
on the instrument that is. Whatever the kl-ladder says, ppl WILL be printed
on the card WITH ITS SIGN (it is expected to be worse, per F76-F78).

THE RECIPE OF RECORD (scratch: overnight_reselect.py = F71 calibration +
referee, assemble_reselect.py = F71 artifact, reselect_35b_param.py /
reselect_bf16_35b.py = F77/F78 arms, flash_reselect_offline.py +
d2_reselect.py = F72/F75 Flash) -- kept here faithfully:

  calibrate
    * the model that GENERATES and is CAPTURED is the VQ ARTIFACT ITSELF
      (overnight_reselect.py: `load(MODEL_ART)`, default art_xtpad), not the
      teacher. "The model will find its own activations" (F68). Kept.
    * 20 plain one-word seeds (below, verbatim), cycled; temp 1.0, no
      top-p/top-k; max 512 new tokens per seed; pool = encode(seed + gen)
      concatenated until >= N tokens, truncated to N (default 8192: F69
      saturation). F70: plain sampling beats steered seeds and a corpus.
    * capture = re-forward the pool in 2048-token chunks (no KV cache across
      chunks) with every VQSwitchLinear's __call__ hooked; per module
      G = sum X^T X over every row the module sees, divided by the row count.
      Full IN x IN Grams are banked (F74's ICM probe needed them); apply
      only ever reads the D x D diagonal blocks.
  apply
    * target W = teacher weights, fp32. F71 used the affine-8bit dequant,
      F77 bf16; F78 found the ppl verdict invariant to that. Either works
      here (`--target` may be an unquantized or an affine-quantized mlx dir).
    * scales RECOMPUTED from the target: max|W| per group + 1e-8, stored in
      the base artifact's vq_scales dtype (the old scripts replaced the
      artifact's scales too -- see "changed" below).
    * per subvector j: code = argmin_c (w_j - c)^T G_jj (w_j - c) over the
      normalized subvector w_j = W_j / scale, codebook unchanged. Computed
      as xGx - 2 xG c^T + cGc, exactly the old expression.
    * codebooks are NOT retuned: F67 measured tables-only at ~+0.5%,
      "nearly worthless". The old scripts never retuned tables either.
    * block-diagonal only: F74 measured full-Gram ICM overfitting
      catastrophically (+56% train, -20% held-out).

WHAT CHANGED vs the scratch scripts (each deliberate):
  * seed: default 1234 (lab rule), -1 = unseeded. The old run used
    mx.random.seed(11); the bank's manifest records which was used.
  * group size is read from the artifact's config (`vq_modules[m].group`),
    not hard-coded 64 (flash_reselect_offline.py hard-coded it).
  * packing uses runtime/vq_pack.pack (bit-identical layout to the old
    inline pack(); unaligned tails are padded). Unpacked (uint8/uint16)
    codes, e.g. Flash d2 layers 0-1 (d2_reselect.py), stay unpacked in
    their own dtype -- one code path for both formats.
  * `--scales keep` is offered (normalize the target by the SHIPPED scales,
    so only codes change). Default `recompute` is the recipe of record.
    The old arms changed codes AND scales at once, so F76-F78 never
    separated a code effect from a scale effect.
  * a module whose codebook/scales/target is missing is an ERROR, not a
    silent skip (assemble_reselect.py skipped silently).
  * the output is a new dir: untouched shards and runtime files are
    symlinked/copied, rewritten shards are new files, and a build record
    (vqlab_provenance.json) names the recipe, the calibration manifest hash
    and the seed. Never in place.
  * the argmin runs on the device you name (`--device cpu|gpu`), per-module
    checkpointed in <out>.parts/ so a GPU-timeout crash resumes (F77 ops).

Scope: MoE expert modules (VQSwitchLinear, [E, OUT, *] codes) -- what the
recipe of record covered. Dense VQLinear is not handled.
Fits are NOT filed into the fit store: re-selected codes reuse the base's
codebook and are not a k-means fit of that recipe; filing them would let
`geo-build --pool` reuse them as one.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import pathlib
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # src/
from vqlab import _layout  # noqa: E402,F401  one module object per name
import vq_pack  # noqa: E402
import provenance  # noqa: E402

SCRATCH = "<scratch>"
SCHEMA = "vqlab.reselect.grams/1"
MANIFEST = "manifest.json"

# overnight_reselect.py SEEDS, verbatim (F69/F70 plain-sampling recipe)
SEEDS = ["The", "In", "A", "One", "When", "It", "We", "After", "Every",
         "def", "import", "class", "if", "for", "Let", "Suppose",
         "Given", "x", "1", "#"]
TEMP = 1.0
MAX_PER_SEED = 512
CAPTURE_CHUNK = 2048
SCALE_EPS = 1e-8
MODULE_CLASS = "VQSwitchLinear"
_AP = None


def _log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def _require_scratch(p):
    rp = os.path.realpath(p)
    if not rp.startswith("/Volumes/"):
        sys.exit(f"REFUSED: {p} is on the internal disk; outputs go under "
                 f"{SCRATCH}/ (AGENTS.md)")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _fname(module):
    return module.replace("/", "_") + ".safetensors"


# ======================================================================= math
def gram_blocks(G, dim):
    """[IN, IN] Gram -> [IN/dim, dim, dim] diagonal blocks."""
    G = np.asarray(G, dtype=np.float32)
    n = G.shape[0] // dim
    idx = np.arange(n)
    return G.reshape(n, dim, n, dim)[idx, :, idx, :]


def reselect_codes(Wn, codebook, blocks, *, rows_per_chunk=None, eval_every=4):
    """G-aware re-selection. Wn: [R, IN] NORMALIZED target rows; codebook
    [K, D]; blocks [IN/D, D, D]. Returns int codes [R, IN/D].

    code[r, j] = argmin_c (w - c)^T G_j (w - c), w = Wn[r, jD:(j+1)D]; the
    old scripts' expression xGx - 2 xG c^T + cGc, computed in fp32 on the
    current MLX default device (mx.set_default_device picks CPU/GPU)."""
    import mlx.core as mx
    R, IN = Wn.shape
    K, D = codebook.shape
    nsub = IN // D
    assert nsub * D == IN and blocks.shape == (nsub, D, D), (Wn.shape, blocks.shape)
    cb = mx.array(np.asarray(codebook, dtype=np.float32))
    Gb = mx.array(np.asarray(blocks, dtype=np.float32))
    rows_per_chunk = rows_per_chunk or R
    out = np.empty((R, nsub), dtype=np.int64)
    for r0 in range(0, R, rows_per_chunk):
        Wc = mx.array(np.asarray(Wn[r0:r0 + rows_per_chunk], dtype=np.float32))
        Wc = Wc.reshape(-1, nsub, D).transpose(1, 0, 2)          # [nsub, r, D]
        mx.eval(Wc)
        cps = []
        for j in range(nsub):
            Gj = Gb[j]
            Xj = Wc[j]
            XG = Xj @ Gj
            CG = cb @ Gj
            d = mx.sum(XG * Xj, 1, keepdims=True) - 2 * (XG @ cb.T) \
                + mx.sum(CG * cb, 1)[None]
            cps.append(mx.argmin(d, axis=1))
            if j % eval_every == 0:      # Metal watchdog: bound lazy argmins
                mx.eval(cps[-1])
        cc = mx.stack(cps, axis=1)
        mx.eval(cc)
        out[r0:r0 + rows_per_chunk] = np.array(cc)
        del Wc, cps, cc
        mx.clear_cache()
    return out


def blockdiag_error(Wn, codes, codebook, blocks):
    """Per-row sum_j e_j^T G_j e_j, e = w - codebook[code] (normalized space).
    Minimized exactly, subvector by subvector, by reselect_codes."""
    R, IN = Wn.shape
    K, D = codebook.shape
    nsub = IN // D
    E = Wn.reshape(R, nsub, D) - codebook[codes]                  # [R,nsub,D]
    return np.einsum("rjd,jde,rje->r", E, blocks, E)


def normalize(W, group, scales=None):
    """W [E, OUT, IN] fp32 -> (Wn [E*OUT, IN], scales [E, OUT, IN/group]).
    scales=None: recompute max|W| per group + 1e-8 (recipe of record)."""
    E, OUT, IN = W.shape
    Wg = W.reshape(E, OUT, IN // group, group)
    if scales is None:
        scales = np.abs(Wg).max(-1) + SCALE_EPS
    Wn = (Wg / scales[..., None]).reshape(E * OUT, IN)
    return Wn.astype(np.float32), scales.astype(np.float32)


# ================================================================ calibrate
def _load_model(path):
    from mlx_lm.utils import load
    import inspect
    kw = {}
    try:
        if "trust_remote_code" in inspect.signature(load).parameters:
            kw["trust_remote_code"] = True
    except (TypeError, ValueError):
        pass
    try:
        return load(str(path), **kw)
    except TypeError:
        if not kw:
            raise
        return load(str(path))


def cmd_calibrate(a):
    cfg = json.load(open(os.path.join(a.model, "config.json")))
    vqm = cfg.get("vq_modules") or cfg.get("text_config", {}).get("vq_modules") or {}
    plan = {
        "schema": SCHEMA, "model": os.path.realpath(a.model),
        "model_role": "base artifact (VQ student): the recipe of record "
                      "generates from and captures the artifact itself (F68)",
        "seed": a.seed if a.seed >= 0 else "unseeded",
        "seeds": SEEDS, "tokens": a.tokens,
        "sampling": {"temp": TEMP, "top_p": None, "top_k": None,
                     "max_tokens_per_seed": MAX_PER_SEED,
                     "sampler": "mlx_lm.sample_utils.make_sampler(temp=1.0)"},
        "capture": {"chunk": CAPTURE_CHUNK, "module_class": MODULE_CLASS,
                    "gram": "full IN x IN, fp32, / rows seen",
                    "rows": "every row the module's __call__ receives "
                            "(gate/up: each token once; down: each routed "
                            "(token, expert) pair, pooled across experts)"},
        "vq_modules_in_config": len(vqm),
        "out": a.out,
    }
    if a.dry_run:
        print("DRY RUN -- nothing loaded, nothing written. Plan:")
        print(json.dumps(plan, indent=1))
        est = sum(int(v["in"]) ** 2 * 4 for v in vqm.values())
        print(f"bank size ~{est / 2**30:.2f} GiB ({len(vqm)} Grams)")
        return 0
    _require_scratch(a.out)
    os.makedirs(a.out, exist_ok=False)

    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    if a.seed >= 0:
        mx.random.seed(a.seed)
    model, tok = _load_model(a.model)
    mods = [(n, m) for n, m in model.named_modules()
            if type(m).__name__ == MODULE_CLASS]
    _log(f"{len(mods)} {MODULE_CLASS} modules")
    if not mods:
        sys.exit("no VQ expert modules found -- wrong artifact?")

    _log(f"self-generating {a.tokens} calibration tokens")
    sampler = make_sampler(temp=TEMP)
    pool, si, used = [], 0, []
    while len(pool) < a.tokens:
        p = SEEDS[si % len(SEEDS)]
        si += 1
        out = "".join(r.text for r in stream_generate(
            model, tok, p, max_tokens=MAX_PER_SEED, sampler=sampler))
        pool += tok.encode(p + out)
        used.append(p)
    ids = pool[:a.tokens]
    np.save(os.path.join(a.out, "tokens.npy"), np.array(ids, dtype=np.int64))

    _log("capturing per-module Grams")
    grams, counts = {}, {}
    cls = type(mods[0][1])
    orig_call = cls.__call__
    name_of = {id(m): n for n, m in mods}

    def hook(self, xx, *args, **kw):
        nm = name_of.get(id(self))
        if nm is not None:
            X = xx.reshape(-1, xx.shape[-1]).astype(mx.float32)
            g = X.T @ X
            grams[nm] = grams[nm] + g if nm in grams else g
            counts[nm] = counts.get(nm, 0) + X.shape[0]
            mx.eval(grams[nm])
        return orig_call(self, xx, *args, **kw)

    cls.__call__ = hook
    try:
        for i in range(0, len(ids), CAPTURE_CHUNK):
            mx.eval(model(mx.array([ids[i:i + CAPTURE_CHUNK]])))
    finally:
        cls.__call__ = orig_call
    gdir = os.path.join(a.out, "grams")
    os.makedirs(gdir)
    for n in sorted(grams):
        mx.save_safetensors(os.path.join(gdir, _fname(n)),
                            {"gram": grams[n] / counts[n]},
                            metadata={"module": n, "rows": str(counts[n])})
    plan.update(seeds_used=used, rows=counts, modules=sorted(grams),
                tokens_sha256=_sha256(os.path.join(a.out, "tokens.npy")),
                created=time.strftime("%Y-%m-%dT%H:%M:%S"),
                code=provenance.code_state(__file__))
    json.dump(plan, open(os.path.join(a.out, MANIFEST), "w"), indent=1)
    _log(f"CALIBRATE-DONE: {len(grams)} Grams -> {gdir}")
    return 0


# ==================================================================== apply
class _Index:
    """tensor name -> real file, over one mlx safetensors dir."""

    def __init__(self, d):
        self.d = d
        ip = os.path.join(d, "model.safetensors.index.json")
        if os.path.exists(ip):
            wm = json.load(open(ip))["weight_map"]
            self.map = {k: os.path.join(d, v) for k, v in wm.items()}
        else:
            import mlx.core as mx
            self.map = {}
            for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
                for k in mx.load(f):
                    self.map[k] = f

    def has(self, k):
        return k in self.map

    def get(self, k):
        """Load on the CPU stream with eval INSIDE the block (FINDINGS IV.1)."""
        import mlx.core as mx
        with mx.stream(mx.cpu):
            v = mx.load(os.path.realpath(self.map[k]))[k]
            mx.eval(v)
        return v


def _target_weight(tix, tcfg, n):
    import mlx.core as mx
    k = n + ".weight"
    if not tix.has(k):
        sys.exit(f"target has no {k} (mlx-format teacher expected, e.g. the "
                 f"mlx-community snapshot)")
    w = tix.get(k)
    if tix.has(n + ".scales"):
        q = (tcfg.get("quantization") or {})
        q = q.get(n, q)
        with mx.stream(mx.cpu):
            w = mx.dequantize(w, tix.get(n + ".scales"), tix.get(n + ".biases"),
                              group_size=int(q.get("group_size", 64)),
                              bits=int(q.get("bits", 8)))
            mx.eval(w)
    with mx.stream(mx.cpu):
        w = w.astype(mx.float32)
        mx.eval(w)
    return np.array(w)


def _method(a, man_sha, man):
    return {"recipe": "G-aware code re-selection (F67-F75): codebook fixed, "
                      "per-subvector argmin of (w-c)^T G_jj (w-c) under the "
                      "BLOCK-DIAGONAL activation Gram (F74: full-Gram overfits); "
                      "no table retune (F67)",
            "gram": "self-generated calibration from the base artifact",
            "calibration_manifest_sha256": man_sha,
            "calibration_seed": man.get("seed"),
            "calibration_tokens": man.get("tokens"),
            "calibration_model": man.get("model"),
            "target": os.path.realpath(a.target),
            "scales": ("recomputed max-abs per group + 1e-8 (recipe of record)"
                       if a.scales == "recompute" else "kept from base artifact"),
            "tables_retuned": False,
            "argmin_device": a.device,
            "why": "re-test under the F118 KL gate; F78 rejected it on ppl, "
                   "which is now printed on the card with its sign, not gated"}


def cmd_apply(a):
    import mlx.core as mx
    mx.set_default_device(mx.cpu if a.device == "cpu" else mx.gpu)
    if os.path.exists(a.out):
        sys.exit(f"REFUSED: {a.out} exists; apply always writes a NEW dir")
    if os.path.realpath(a.out) == os.path.realpath(a.artifact):
        sys.exit("REFUSED: never in place")
    if not a.allow_any_out:
        _require_scratch(a.out)
    man_path = os.path.join(a.grams, MANIFEST)
    man = json.load(open(man_path))
    assert man.get("schema") == SCHEMA, man.get("schema")
    man_sha = _sha256(man_path)

    cfg = json.load(open(os.path.join(a.artifact, "config.json")))
    vqm = cfg.get("vq_modules") or cfg.get("text_config", {}).get("vq_modules")
    tcfg = json.load(open(os.path.join(a.target, "config.json")))
    aix, tix = _Index(a.artifact), _Index(a.target)
    mods = sorted(man["modules"])
    if a.modules:
        want = set(a.modules.split(","))
        mods = [m for m in mods if m in want]
    missing = [m for m in mods if m not in vqm or not aix.has(m + ".codes")]
    if missing:
        sys.exit(f"{len(missing)} banked modules absent from the artifact, e.g. {missing[:3]}")

    parts = a.out.rstrip("/") + ".parts"
    os.makedirs(parts, exist_ok=True)
    report = {}
    t0 = time.time()
    for i, n in enumerate(mods):
        part = os.path.join(parts, _fname(n))
        rep = os.path.join(parts, _fname(n) + ".json")
        if os.path.exists(part) and os.path.exists(rep):
            report[n] = json.load(open(rep))
            continue
        cb = np.array(aix.get(n + ".codebook").astype(mx.float32))
        K, D = cb.shape
        group = int(vqm[n].get("group", 64))
        base_codes = aix.get(n + ".codes")
        base_scales = aix.get(n + ".vq_scales")
        W = _target_weight(tix, tcfg, n)
        E, OUT, IN = W.shape
        nsub = IN // D
        Wn, scl = normalize(W, group, None if a.scales == "recompute"
                            else np.array(base_scales.astype(mx.float32)))
        del W
        G = np.array(mx.load(os.path.join(a.grams, "grams", _fname(n)))["gram"])
        assert G.shape == (IN, IN), (n, G.shape, IN)
        blocks = gram_blocks(G, D)
        del G
        codes = reselect_codes(Wn, cb, blocks, rows_per_chunk=a.row_chunk * OUT)
        bc = np.array(base_codes)
        if base_codes.dtype == mx.uint32:
            bits = vq_pack.bits_for_k(K)
            old = vq_pack.unpack(bc, nsub, bits).reshape(E * OUT, nsub)
            new_t = vq_pack.pack(codes.reshape(E, OUT, nsub), bits)
        else:
            old = bc.reshape(E * OUT, nsub)
            new_t = codes.reshape(E, OUT, nsub).astype(bc.dtype)
        assert new_t.shape == bc.shape, (n, new_t.shape, bc.shape)
        j_old = float(blockdiag_error(Wn, old.astype(np.int64), cb, blocks).sum())
        j_new = float(blockdiag_error(Wn, codes, cb, blocks).sum())
        r = {"dim": int(D), "k": int(K), "origin": "reselect",
             "changed_frac": float((old != codes).mean()),
             "J_train_base_codes": j_old, "J_train_new": j_new,
             "J_train_gain": (j_old - j_new) / j_old if j_old else 0.0}
        mx.save_safetensors(part, {
            n + ".codes": mx.array(new_t),
            n + ".vq_scales": mx.array(scl).astype(base_scales.dtype)})
        json.dump(r, open(rep, "w"))
        report[n] = r
        del Wn, blocks, codes
        mx.clear_cache()
        _log(f"[{i + 1}/{len(mods)}] {n}: {100 * r['changed_frac']:.1f}% codes changed, "
             f"J_train {100 * r['J_train_gain']:+.2f}% [{time.time() - t0:.0f}s]")

    # ---- assemble a NEW dir: rewritten shards new, everything else linked
    new = {}
    for n in mods:
        new.update(mx.load(os.path.join(parts, _fname(n))))
    os.makedirs(a.out)
    touched = {aix.map[k] for k in new}
    for f in sorted(glob.glob(os.path.join(a.artifact, "*"))):
        b = os.path.basename(f)
        if b == "__pycache__" or b == provenance.RECORD or os.path.isdir(f):
            continue
        dst = os.path.join(a.out, b)
        if b.endswith(".safetensors"):
            if f in touched:
                tens = mx.load(os.path.realpath(f))
                out = {}
                for k, v in tens.items():
                    if k in new:
                        assert new[k].shape == v.shape and new[k].dtype == v.dtype, k
                        out[k] = new[k]
                    else:
                        out[k] = v
                mx.save_safetensors(dst, out, metadata={"format": "mlx"})
                del tens, out
                mx.clear_cache()
            else:
                os.symlink(os.path.realpath(f), dst)
        else:
            shutil.copy(os.path.realpath(f), dst)   # config/model.py: real copies
    json.dump(report, open(os.path.join(a.out, "reselect_report.json"), "w"), indent=1)
    provenance.write_build_record(
        a.out, tool="reselect", script=__file__, ap=_AP, args=a,
        method=_method(a, man_sha, man),
        inputs=[("base", a.artifact), ("teacher", a.target),
                ("calibration", man_path)],
        modules=report,
        full_hash={os.path.basename(p) for p in glob.glob(os.path.join(a.out, "*"))
                   if not os.path.islink(p)})
    gains = [r["J_train_gain"] for r in report.values()]
    _log(f"APPLY-DONE: {len(report)} modules, median J_train gain "
         f"{100 * float(np.median(gains)):+.2f}% -> {a.out}")
    return 0


# ================================================================= selftest
def selftest(root=None):
    """Synthetic, CPU-only: a random 2-expert module, a codebook, a Gram
    with a planted anisotropic structure. Checks (1) re-selection never
    raises and on the whole lowers the training block-diagonal error vs
    nearest-neighbour (k-means) codes; (2) `apply` writes a NEW dir with a
    build record and leaves the base untouched. Returns the out dir."""
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    rng = np.random.default_rng(1234)
    root = root or os.path.join(SCRATCH, "reselect_selftest",
                                time.strftime("%Y%m%d-%H%M%S"))
    E, OUT, IN, D, K, group = 2, 16, 64, 4, 16, 32
    nsub = IN // D
    name = "language_model.model.layers.0.mlp.switch_mlp.gate_proj"
    W = rng.standard_normal((E, OUT, IN)).astype(np.float32)
    Wn, scl = normalize(W, group)
    cb = rng.standard_normal((K, D)).astype(np.float32) * 0.5
    km = np.argmin(((Wn.reshape(-1, nsub, 1, D) - cb) ** 2).sum(-1), -1)   # k-means assign
    A = rng.standard_normal((IN, IN)).astype(np.float32)
    A[:, ::D] *= 8.0                                   # planted high-energy directions
    G = (A.T @ A / IN).astype(np.float32)
    blocks = gram_blocks(G, D)
    new = reselect_codes(Wn, cb, blocks)
    e_km = blockdiag_error(Wn, km, cb, blocks)
    e_new = blockdiag_error(Wn, new, cb, blocks)
    assert np.all(e_new <= e_km + 1e-5 * np.abs(e_km)), "re-selection raised error"
    assert e_new.sum() < 0.99 * e_km.sum(), (e_new.sum(), e_km.sum())

    art, tgt, bank = (os.path.join(root, x) for x in ("base", "teacher", "bank"))
    for d in (art, tgt, os.path.join(bank, "grams")):
        os.makedirs(d)
    bits = vq_pack.bits_for_k(K)
    vqm = {name: {"experts": E, "out": OUT, "in": IN, "k": K, "dim": D,
                  "group": group, "pack_bits": bits}}
    json.dump({"model_type": "synthetic", "vq_modules": vqm},
              open(os.path.join(art, "config.json"), "w"))
    json.dump({"model_type": "synthetic"}, open(os.path.join(tgt, "config.json"), "w"))
    mx.save_safetensors(os.path.join(art, "model.safetensors"), {
        name + ".codes": mx.array(vq_pack.pack(km.reshape(E, OUT, nsub), bits)),
        name + ".codebook": mx.array(cb).astype(mx.float16),
        name + ".vq_scales": mx.array(scl).astype(mx.float16)})
    cb = np.array(mx.array(cb).astype(mx.float16).astype(mx.float32))  # as stored
    mx.save_safetensors(os.path.join(tgt, "model.safetensors"),
                        {name + ".weight": mx.array(W)})
    mx.save_safetensors(os.path.join(bank, "grams", _fname(name)), {"gram": mx.array(G)})
    json.dump({"schema": SCHEMA, "modules": [name], "seed": 1234, "tokens": 0,
               "model": "synthetic"}, open(os.path.join(bank, MANIFEST), "w"))
    base_before = _sha256(os.path.join(art, "model.safetensors"))
    out = os.path.join(root, "out")
    rc = main(["apply", "--artifact", art, "--grams", bank, "--target", tgt,
               "--out", out, "--device", "cpu"])
    assert rc == 0
    assert _sha256(os.path.join(art, "model.safetensors")) == base_before, "base edited"
    rec = provenance.load(out)
    assert rec["tool"]["name"] == "reselect"
    assert rec["method"]["calibration_manifest_sha256"] == _sha256(os.path.join(bank, MANIFEST))
    r = rec["modules"][name]
    assert r["J_train_new"] <= r["J_train_base_codes"] and r["J_train_gain"] > 0.01, r
    got = vq_pack.unpack(np.array(mx.load(os.path.join(out, "model.safetensors"))[name + ".codes"]),
                         nsub, bits).reshape(-1, nsub)
    assert np.array_equal(got, reselect_codes(Wn, cb, blocks)), "stored codes != re-selected"
    print(f"SELFTEST OK: J_train {e_km.sum():.3f} -> {e_new.sum():.3f} "
          f"({100 * (1 - e_new.sum() / e_km.sum()):.1f}% lower); apply gain "
          f"{100 * r['J_train_gain']:.1f}%; new dir {out}")
    return out


# ===================================================================== main
def main(argv=None):
    ap = argparse.ArgumentParser(prog="vqlab reselect",
                                 description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate", help="self-generate from the base artifact, bank block Grams")
    c.add_argument("--model", required=True,
                   help="the VQ artifact to generate from and capture (recipe of record: the base artifact itself)")
    c.add_argument("--out", required=True, help="new Gram-bank dir (under the SSD scratch)")
    c.add_argument("--tokens", type=int, default=8192, help="calibration tokens (F69: saturates by 8k)")
    c.add_argument("--seed", type=int, default=1234, help="sampling seed; -1 = unseeded")
    c.add_argument("--dry-run", action="store_true", help="print the plan, load nothing")
    p = sub.add_parser("apply", help="re-select codes into a NEW artifact dir")
    p.add_argument("--artifact", required=True, help="base VQ artifact (never edited)")
    p.add_argument("--grams", required=True, help="bank dir from `calibrate`")
    p.add_argument("--target", required=True, help="teacher weights (mlx dir; bf16 or affine)")
    p.add_argument("--out", required=True, help="new artifact dir (must not exist)")
    p.add_argument("--scales", choices=["recompute", "keep"], default="recompute",
                   help="recompute = recipe of record; keep = shipped scales, codes-only")
    p.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    p.add_argument("--row-chunk", type=int, default=8, help="experts per argmin chunk")
    p.add_argument("--modules", default=None, help="comma list: only these modules")
    p.add_argument("--allow-any-out", action="store_true", help=argparse.SUPPRESS)
    s = sub.add_parser("selftest", help="CPU synthetic check (no model, no GPU)")
    s.add_argument("--dir", default=None)
    a = ap.parse_args(argv)
    global _AP
    _AP = ap
    if a.cmd == "calibrate":
        return cmd_calibrate(a)
    if a.cmd == "apply":
        return cmd_apply(a)
    selftest(a.dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
