"""sz-bitexact (EXPERIMENTAL): BYTE-equality gate for vq-skipzero stage 2.

    vqlab sz-bitexact <resident> <reference> [--ref2 <stage1_pack>]
                      [--mem-ref <stage1_pack>] --out <dir under vqlab-scratch>
    vqlab sz-bitexact --synthetic --out <dir> [--vintage walk|u8|both|native|all]
                      [--u8-model-py <397B bundle model.py>]   # GPU kernel stress, no model
    vqlab sz-bitexact --selftest                 # CPU only, no GPU

Model mode runs ONE subprocess per artifact (a clean process each: separate
kernel caches, clean memory counters), each loading with mlx_lm on the GPU,
and dumps raw output bytes; the parent compares them AS BYTES (max abs diff
printed only to locate a failure -- any unequal byte is a FAIL):
  * module level: every swapped switch module, fed identical seeded inputs
    (experts biased to the dead-row-heavy ones), at N = 1, 8, 4096 (fused
    decode kernel) and 4097 (gemmseg prefill kernel) token-expert pairs;
  * in-model module outputs of every swapped module on a real 1-token and a
    513-token (N=4104, prefill path) forward;
  * final logits at T = 1, 512 (N=4096), 513; a ~9k-token prefill in 2048
    chunks through the KV cache (per-chunk full-logit sha256 + last-row
    bytes); a 32-token greedy generation (token ids + every step's logits).
--mem-ref additionally loads a stage-1 pack only to report its memory.
Reference = the ORIGINAL rung on the 35B (F176); --ref2 adds a second one.

Two runtime vintages (sz_resident.vintage): "walk" (repo runtime, 35B; packed
uint32) and "u8" (the 397B bundle; unpacked uint8 d4 K256). --synthetic
compares each vintage's SZ kernels against THAT vintage's own kernels on the
expanded tensors: walk against runtime/vq_switch.py, u8 against the runtime
part of the 397B bundle's model.py (exec'd from its text, read-only).
--vintage native stresses the runtime's own row-table switch (packed d4,
packed d2 at K256-K2048, and the 397B 2.2's packed d8 K16384) against the
same runtime on expanded tensors; --vintage all runs all three.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parents[1]
sys.path.insert(0, str(SRC))
from vqlab import config  # noqa: E402

SEED = 1234
# The last runtime/vq_switch.py without the native SZ switch, shipped as package data
WALK_FROZEN = SRC / "vqlab" / "runtime" / "equivalent" / "vq_switch.40870875499d.py.txt"
# A u8-vintage bundle (the published 397B 2.4bpw runtime), under config.models()
U8_MODEL_PY = config.models() / "TheDrainFlorist--Qwen3.5-397B-A17B-VQ-2.4bpw" / "model.py"
NCASES = {1: (1, 1), 8: (1, 8), 4096: (512, 8), 4097: (4097, 1)}   # N: (T, k)
CORPUS = SRC / "vqlab" / "score" / "referee" / "referee_corpus.txt"


def _log(*a):
    print("[sz-bitexact]", *a, flush=True)


# ------------------------------------------------------------ shared helpers
def _to_np(a):
    """mx array -> numpy with the SAME bytes (bf16 viewed as uint16)."""
    import mlx.core as mx
    mx.eval(a)
    if a.dtype == mx.bfloat16:
        a = mx.view(a, mx.uint16)
    return np.array(a, copy=True)


def _as_f32(a, dt):
    if dt == "bfloat16":
        return (a.astype(np.uint32) << 16).view(np.float32)
    return a.astype(np.float32)


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def dead_experts(art):
    """{module: experts ordered by dead-row count desc} from the resident pack."""
    sys.path.insert(0, str(SRC))
    from vqlab import _layout  # noqa: F401
    import fitstore
    cfg = json.loads((pathlib.Path(art) / "config.json").read_text())
    mods = cfg["vq_skipzero"]["modules"]
    out = {}
    for f in sorted(pathlib.Path(art).glob("*.safetensors")):
        h, base = fitstore.read_header(f)
        for k in h:
            if k.endswith(".sz_rowmask") and k[:-11] in mods:
                p = k[:-11]
                E, OUT = mods[p]["experts"], mods[p]["out"]
                bits = np.unpackbits(np.frombuffer(fitstore.read_tensor_bytes(f, h, base, k),
                                                   np.uint8).reshape(E, -1), axis=-1,
                                     bitorder="little")[:, :OUT]
                dead = OUT - bits.sum(axis=1)
                out[p] = [int(e) for e in np.argsort(-dead, kind="stable")]
    return out


def module_inputs(p, i, n, IN, E, order, is_down):
    T, k = NCASES[n]
    rng = np.random.default_rng([SEED, i, n])
    hot = order[: max(1, min(16, E))]
    idx = np.where(rng.random((T, k)) < 0.5, rng.choice(hot, (T, k)),
                   rng.integers(0, E, (T, k))).astype(np.uint32)
    xs = (T, k, 1, IN) if is_down else (T, 1, 1, IN)
    x = (rng.standard_normal(xs) * 0.5).astype(np.float32)
    return x, idx


# ------------------------------------------------------------------- worker
def worker(art, role, outdir, experts_json, mem_only):
    import mlx.core as mx
    from mlx_lm.utils import load
    from mlx_lm.models.cache import make_prompt_cache
    out = pathlib.Path(outdir) / role
    out.mkdir(parents=True, exist_ok=True)
    rep = {"artifact": str(art), "role": role,
           "env": {k: v for k, v in os.environ.items() if k.startswith("VQ")},
           "model_py_mtime": os.stat(pathlib.Path(art) / "model.py").st_mtime}
    mx.reset_peak_memory()
    t0 = time.time()
    model, tok = load(str(art))
    mx.eval(model.parameters())
    rep["load_s"] = time.time() - t0
    rep["active_gib"] = mx.get_active_memory() / 2**30
    rep["peak_gib"] = mx.get_peak_memory() / 2**30
    modns = type(model).__init__.__globals__   # mlx_lm loads model.py without registering it in sys.modules
    rep["resident_classes"] = sorted({type(m).__name__ for _, m in model.named_modules()
                                      if "Switch" in type(m).__name__})
    if mem_only:
        json.dump(rep, open(out / "report.json", "w"), indent=1)
        return 0
    order = json.load(open(experts_json))
    sys.path.insert(0, str(HERE))
    import sz_resident
    reach = modns.get("_reach_vq") or sz_resident.reach   # the 397B bundle has no _reach_vq
    mods = list(order)
    cfg = json.loads((pathlib.Path(art) / "config.json").read_text())

    # 1. module level, seeded identical inputs
    for i, p in enumerate(mods):
        obj, leaf = reach(model, p)
        m = getattr(obj, leaf)
        g = cfg["vq_modules"][p]
        for n in NCASES:
            x, idx = module_inputs(p, i, n, g["in"], g["experts"], order[p],
                                   p.endswith("down_proj"))
            y = m(mx.array(x).astype(mx.bfloat16), mx.array(idx))
            np.save(out / f"mod{i:03d}_N{n}.npy", _to_np(y))
            mx.clear_cache()

    # 2. in-model module outputs (recorder on the class __call__)
    ids = tok.encode(CORPUS.read_text())
    while len(ids) < 9300:
        ids = ids + ids
    rec = {}
    pathmap = {}
    for i, p in enumerate(mods):
        obj, leaf = reach(model, p)
        pathmap[id(getattr(obj, leaf))] = i
    classes = {type(getattr(*reach(model, p))) for p in mods}
    saved = {}
    for c in classes:
        saved[c] = c.__call__

        def wrapped(self, *a, _orig=c.__call__, **kw):
            y = _orig(self, *a, **kw)
            j = pathmap.get(id(self))
            if j is not None:
                rec[j] = _to_np(y)
            return y
        c.__call__ = wrapped
    for T in (1, 513):
        rec.clear()
        mx.eval(model(mx.array(ids[:T])[None]))
        for j, a in rec.items():
            np.save(out / f"inmodel_T{T}_mod{j:03d}.npy", a)
    for c, f in saved.items():
        c.__call__ = f
    rec.clear()

    # 3. logits
    for T in (1, 512, 513):
        np.save(out / f"logits_T{T}.npy", _to_np(model(mx.array(ids[:T])[None])))
        mx.clear_cache()
    cache = make_prompt_cache(model)
    pre = ids[:9216]
    shas, lasts = [], []
    for s in range(0, len(pre), 2048):
        lg = model(mx.array(pre[s:s + 2048])[None], cache=cache)
        a = _to_np(lg)
        shas.append(_sha(a))
        lasts.append(a[0, -1])
        del lg, a
        mx.clear_cache()
    np.save(out / "prefill9k_last.npy", np.stack(lasts))
    rep["prefill9k_chunk_sha"] = shas
    del cache
    cache = make_prompt_cache(model)
    y = model(mx.array(ids[:64])[None], cache=cache)
    toks, steps = [], []
    for _ in range(32):
        last = y[0, -1]
        steps.append(_to_np(last))
        t = int(mx.argmax(last).item())
        toks.append(t)
        y = model(mx.array([[t]]), cache=cache)
    np.save(out / "greedy32_logits.npy", np.stack(steps))
    rep["greedy32_tokens"] = toks
    rep["model_py_mtime_end"] = os.stat(pathlib.Path(art) / "model.py").st_mtime
    json.dump(rep, open(out / "report.json", "w"), indent=1)
    return 0


# ------------------------------------------------------------------ compare
def compare(outdir, a, b, dtype="bfloat16"):
    da, db = pathlib.Path(outdir) / a, pathlib.Path(outdir) / b
    ra, rb = json.load(open(da / "report.json")), json.load(open(db / "report.json"))
    rows, fails = {}, 0
    for f in sorted(da.glob("*.npy")):
        g = db / f.name
        if not g.exists():
            rows.setdefault(f.name, ("MISSING", None)); fails += 1
            continue
        x, y = np.load(f), np.load(g)
        eq = x.shape == y.shape and x.tobytes() == y.tobytes()
        md = None
        if not eq:
            fails += 1
            md = (float(np.nanmax(np.abs(_as_f32(x, dtype) - _as_f32(y, dtype))))
                  if x.shape == y.shape else "shape")
        stem = f.name[:-4]
        if stem.startswith("mod"):
            cat = "module " + stem.split("_")[1]                 # module N<n>
        elif stem.startswith("inmodel"):
            cat = "in-model " + stem.split("_")[1]               # in-model T<t>
        else:
            cat = stem
        c = rows.setdefault(cat, [0, 0, 0.0])
        c[0] += 1
        c[1] += int(eq)
        if md is not None and md != "shape":
            c[2] = max(c[2], md)
    ext = {"prefill9k_chunk_sha": ra.get("prefill9k_chunk_sha") == rb.get("prefill9k_chunk_sha"),
           "greedy32_tokens": ra.get("greedy32_tokens") == rb.get("greedy32_tokens")}
    fails += sum(not v for v in ext.values())
    print(f"\n== {a} vs {b} ==")
    print(f"{'check':<24}{'equal':>12}  max|diff|")
    for k, v in rows.items():
        if isinstance(v, tuple):
            print(f"{k:<24}{'MISSING':>12}")
        else:
            print(f"{k:<24}{v[1]:>6}/{v[0]:<5}  {v[2]:.3e}" if v[1] < v[0] else
                  f"{k:<24}{v[1]:>6}/{v[0]:<5}  0 (bytes equal)")
    for k, v in ext.items():
        print(f"{k:<24}{'yes' if v else 'NO':>12}")
    return fails


def mem_table(outdir, roles):
    print(f"\n{'role':<12}{'active GiB':>12}{'peak GiB':>12}{'load s':>9}  classes")
    for r in roles:
        f = pathlib.Path(outdir) / r / "report.json"
        if f.exists():
            j = json.load(open(f))
            print(f"{r:<12}{j['active_gib']:>12.4f}{j['peak_gib']:>12.4f}{j['load_s']:>9.1f}  "
                  f"{j['resident_classes']}")


# --------------------------------------------------------- synthetic (GPU)
def load_u8_runtime(path=U8_MODEL_PY):
    """The 397B bundle's vq_switch runtime, exec'd from the FILE TEXT up to its
    arch shim (which needs config.json next to it). Read-only; never imported
    as the bundle, never written."""
    text = pathlib.Path(path).read_text()
    marker = "\nimport importlib as _importlib\n"
    if text.count(marker) != 1:
        raise RuntimeError(f"{path}: arch-shim marker found {text.count(marker)}x (need 1)")
    ns = {"__name__": "vq_switch_u8_ro", "__file__": str(path)}
    exec(compile(text[: text.index(marker)], str(path), "exec"), ns)
    return ns


def _live_pattern(rng, E, OUT):
    live = rng.random((E, OUT)) > 0.6                  # ~60% dead scattered
    live[1] = False                                    # a fully dead expert
    live[2, :64] = False                               # a whole OT2 pair of tiles
    live[3, 32:64] = False                             # the second block of a pair
    live[4, OUT - 5:] = False                          # ragged tail
    live[5] = True                                     # a fully live expert
    return live


def _synthetic_cases(name, ref, cmp_, rng, live, E, IN, maxn, lines):
    import mlx.core as mx
    fails = 0
    for N, (T, k) in {1: (1, 1), 8: (1, 8), 64: (8, 8), 4096: (512, 8),
                      4097: (4097, 1), 9000: (1125, 8)}.items():
        x = mx.array((rng.standard_normal((T, k, 1, IN)) * 0.5).astype(np.float32)).astype(mx.bfloat16)
        idx = mx.array(rng.integers(0, E, (T, k)).astype(np.uint32))
        a, b = _to_np(ref(x, idx)), _to_np(cmp_(x, idx))
        eq = a.tobytes() == b.tobytes()
        dead_out = int((~live[np.array(idx)]).sum())
        md = 0.0 if eq else float(np.max(np.abs(_as_f32(a, "bfloat16") - _as_f32(b, "bfloat16"))))
        path = "decode" if N <= maxn else "prefill"
        lines.append(f"{name:<14} N={N:<5} {path:<8} dead outputs {dead_out:>8}  "
                     f"{'EQUAL' if eq else 'DIFF'}  max|diff| {md:.3e}")
        fails += int(not eq)
    return fails


def synthetic(outdir, which="both", u8_model_py=U8_MODEL_PY):
    """Many-dead-rows stress: compact SZ kernels vs each vintage's OWN kernels
    on the expanded tensors, byte for byte, decode and prefill paths."""
    import importlib.util
    import mlx.core as mx
    sys.path.insert(0, str(HERE))
    import sz_resident
    rng = np.random.default_rng(SEED)
    fails, lines = 0, []
    if which in ("walk", "both", "all"):
        # The walk fork targets the last runtime WITHOUT the native switch
        # (the current one no longer takes the patch; see selftest).
        import types
        vs = types.ModuleType("vq_switch_walk_frozen")
        exec(compile(WALK_FROZEN.read_text(), vs.__name__, "exec"), vs.__dict__)
        SZ = sz_resident.install(vars(vs))
        assert SZ.vintage == "walk", SZ.vintage
        K, BITS = 2048, 11
        for name, E, OUT, IN in (("walk gate", 16, 512, 2048), ("walk down", 16, 2048, 512)):
            W = (IN // 4 + 31) // 32 * BITS
            G = IN // 64
            codes = rng.integers(0, 2**32, (E, OUT, W), dtype=np.uint64).astype(np.uint32)
            scales = (rng.standard_normal((E, OUT, G)) * 0.02).astype(np.float16)
            cb = (rng.standard_normal((K, 4))).astype(np.float16)
            live = _live_pattern(rng, E, OUT)
            full_c, full_s = codes.copy(), scales.copy()
            full_c[~live] = 0
            full_s[~live] = 0
            tbl = sz_resident.row_table_np(np.packbits(live, axis=-1, bitorder="little"), E, OUT)
            ref = vs.VQSwitchLinear(mx.array(full_c), mx.array(cb), mx.array(full_s),
                                    pack_bits=BITS, in_features=IN)
            cmp_ = SZ(mx.array(codes.reshape(E * OUT, W)[live.reshape(-1)]), mx.array(cb),
                      mx.array(scales.reshape(E * OUT, G)[live.reshape(-1)]),
                      mx.array(tbl), pack_bits=BITS, in_features=IN)
            fails += _synthetic_cases(name, ref, cmp_, rng, live, E, IN, vs.VQ_FUSED_MAX_N, lines)
    if which in ("native", "all"):
        # The runtime's NATIVE switch (VQSwitchLinear(row_table=...)) against
        # the same runtime on expanded tensors: packed d4 K2048/11 and the
        # 397B 2.2's packed d8 K16384/14, at its gate/up and down shapes,
        # and packed d2 at every K the fleet ships (K256-K2048).
        spec = importlib.util.spec_from_file_location("vq_switch_ro", SRC / "vqlab/runtime/vq_switch.py")
        vs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vs)
        sys.path.insert(0, str(SRC / "vqlab/runtime"))
        import vq_pack
        for name, D, K, BITS, E, OUT, IN in (
                ("native d4 gate", 4, 2048, 11, 16, 512, 2048),
                ("native d8 gate", 8, 16384, 14, 16, 1024, 4096),
                ("native d8 down", 8, 16384, 14, 16, 4096, 1024),
                *((f"native d2 K{k} {s}", 2, k, k.bit_length() - 1, 16, o, i)
                  for k in (256, 512, 1024, 2048)
                  for s, o, i in (("gate", 512, 2048), ("down", 2048, 512)))):
            G = IN // 64
            codes = vq_pack.pack(rng.integers(0, K, (E, OUT, IN // D)).astype(np.uint32), BITS)
            W = codes.shape[-1]
            scales = (rng.standard_normal((E, OUT, G)) * 0.02).astype(np.float16)
            cb = (rng.standard_normal((K, D))).astype(np.float16)
            live = _live_pattern(rng, E, OUT)
            full_c, full_s = codes.copy(), scales.copy()
            full_c[~live] = 0
            full_s[~live] = 0
            tbl = np.full((E, OUT), -1, np.int32)
            tbl[live] = np.arange(int(live.sum()), dtype=np.int32)
            ref = vs.VQSwitchLinear(mx.array(full_c), mx.array(cb), mx.array(full_s),
                                    pack_bits=BITS, in_features=IN)
            cmp_ = vs.VQSwitchLinear(mx.array(codes.reshape(E * OUT, W)[live.reshape(-1)]),
                                     mx.array(cb),
                                     mx.array(scales.reshape(E * OUT, G)[live.reshape(-1)]),
                                     pack_bits=BITS, in_features=IN, row_table=mx.array(tbl))
            fails += _synthetic_cases(name, ref, cmp_, rng, live, E, IN, vs.VQ_FUSED_MAX_N, lines)
    if which in ("u8", "both", "all"):
        ns = load_u8_runtime(u8_model_py)
        SZ = sz_resident.install(ns)
        assert SZ.vintage == "u8", SZ.vintage
        K = 256
        # 397B shapes (gate/up 4096->1024, down 1024->4096) at small E
        for name, E, OUT, IN in (("u8 gate(397B)", 16, 1024, 4096), ("u8 down(397B)", 16, 4096, 1024)):
            NSUB, G = IN // 4, IN // 64
            codes = rng.integers(0, 256, (E, OUT, NSUB)).astype(np.uint8)
            scales = (rng.standard_normal((E, OUT, G)) * 0.02).astype(np.float16)
            cb = (rng.standard_normal((K, 4))).astype(np.float16)
            live = _live_pattern(rng, E, OUT)
            full_c, full_s = codes.copy(), scales.copy()
            full_c[~live] = 0
            full_s[~live] = 0
            tbl = sz_resident.row_table_np(np.packbits(live, axis=-1, bitorder="little"), E, OUT)
            ref = ns["VQSwitchLinear"](mx.array(full_c), mx.array(cb), mx.array(full_s),
                                       group_size=64)
            cmp_ = SZ(mx.array(codes.reshape(E * OUT, NSUB)[live.reshape(-1)]), mx.array(cb),
                      mx.array(scales.reshape(E * OUT, G)[live.reshape(-1)]),
                      mx.array(tbl), group_size=64)
            fails += _synthetic_cases(name, ref, cmp_, rng, live, E, IN, ns["VQ_FUSED_MAX_N"], lines)
    txt = "\n".join(lines)
    print(txt)
    if outdir:
        pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
        (pathlib.Path(outdir) / f"synthetic_{which}.txt").write_text(txt + f"\nfails={fails}\n")
    return 1 if fails else 0


# ------------------------------------------------- partial real weights (GPU)
def partial(resident, stage1, layers, outdir, original=None):
    """A few REAL switch modules, no full model (for boxes that cannot hold it
    resident): per module, the resident path (resident_weights + SZSwitchLinear
    over the compact shard tensors) vs the stage-1 path (skipzero_load
    expansion + the bundle's OWN VQSwitchLinear), at N = 1, 8, 4096, 4097,
    compared as bytes. The runtime is the resident artifact's model.py text
    (exec'd up to its arch shim); sz_resident.py is the artifact's own copy.
    --original adds the ORIGINAL rung's modules as information only."""
    import importlib.util
    import mlx.core as mx
    sys.path.insert(0, str(SRC))
    from vqlab import _layout  # noqa: F401
    import fitstore
    res, st1 = pathlib.Path(resident), pathlib.Path(stage1)
    cfg = json.loads((res / "config.json").read_text())
    sz = cfg["vq_skipzero"]
    if not sz.get("resident"):
        raise SystemExit(f"{res} is not a resident artifact")
    ns = load_u8_runtime(res / "model.py") if sz.get("runtime_vintage") == "u8" else None
    if ns is None:
        raise SystemExit("--partial: only the u8 vintage bundle is wired (walk: use model mode)")
    spec = importlib.util.spec_from_file_location("sz_resident_art", res / "sz_resident.py")
    szr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(szr)
    SZ = szr.install(ns)
    sys.path.insert(0, str(HERE))
    import skipzero_load
    lo, hi = (int(v) for v in layers.split("-"))
    mods = [p for p in sz["modules"]
            if lo <= int(p.split(".layers.")[1].split(".")[0]) <= hi]

    def tensors(art, keys):
        idx = json.loads((pathlib.Path(art) / "model.safetensors.index.json").read_text())["weight_map"]
        out = {}
        for k in keys:
            f = pathlib.Path(art) / idx[k]
            h, base = fitstore.read_header(f)
            dt = {"U8": np.uint8, "U32": np.uint32, "F16": np.float16, "I32": np.int32}[h[k]["dtype"]]
            out[k] = np.frombuffer(fitstore.read_tensor_bytes(f, h, base, k), dt).reshape(h[k]["shape"])
        return out

    lines, fails = [], 0
    for i, p in enumerate(mods):
        g = cfg["vq_modules"][p]
        E, OUT, IN = g["experts"], g["out"], g["in"]
        keys = [p + s for s in (".sz_shape", ".sz_rowmask", ".sz_codes", ".sz_scales", ".codebook")]
        t = tensors(res, keys)
        t1 = tensors(st1, keys)
        same_disk = all(t[k].tobytes() == t1[k].tobytes() for k in keys)
        # resident: the artifact's own resident_weights on the shard tensors
        w = szr.resident_weights({k: mx.array(v) for k, v in t.items()})
        m_res = SZ(w[p + ".codes"], w[p + ".codebook"], w[p + ".vq_scales"],
                   w[p + ".row_table"], group_size=g["group"])
        # stage 1: the stage-1 expansion (mlx, as its loader does) + bundle VQSwitchLinear
        full_c = skipzero_load.expand_mx(t1[p + ".sz_shape"], mx.array(t1[p + ".sz_rowmask"]),
                                         mx.array(t1[p + ".sz_codes"]))
        full_s = skipzero_load.expand_mx(t1[p + ".sz_shape"], mx.array(t1[p + ".sz_rowmask"]),
                                         mx.array(t1[p + ".sz_scales"]))
        m_st1 = ns["VQSwitchLinear"](full_c, mx.array(t1[p + ".codebook"]), full_s,
                                     group_size=g["group"])
        m_org = None
        if original:
            to = tensors(original, [p + ".codes", p + ".vq_scales", p + ".codebook"])
            m_org = ns["VQSwitchLinear"](mx.array(to[p + ".codes"]), mx.array(to[p + ".codebook"]),
                                         mx.array(to[p + ".vq_scales"]), group_size=g["group"])
        live = skipzero_load.live_mask_np(t[p + ".sz_rowmask"], E, OUT)
        order = [int(e) for e in np.argsort(-(~live).sum(1), kind="stable")]
        for n in NCASES:
            x, idx = module_inputs(p, i, n, IN, E, order, p.endswith("down_proj"))
            xm, im = mx.array(x).astype(mx.bfloat16), mx.array(idx)
            a, b = _to_np(m_res(xm, im)), _to_np(m_st1(xm, im))
            eq = a.tobytes() == b.tobytes()
            md = 0.0 if eq else float(np.max(np.abs(_as_f32(a, "bfloat16") - _as_f32(b, "bfloat16"))))
            dead_out = int((~live[idx.reshape(-1)]).sum())
            info = ""
            if m_org is not None:
                c = _to_np(m_org(xm, im))
                info = "  vs original: " + ("EQUAL" if c.tobytes() == a.tobytes() else
                        f"diff {np.max(np.abs(_as_f32(a, 'bfloat16') - _as_f32(c, 'bfloat16'))):.2e}")
            lines.append(f"{p.split('model.')[-1]:<38} N={n:<5} dead outputs {dead_out:>8}/"
                         f"{idx.size * OUT:<8} {'EQUAL' if eq else 'DIFF'} max|diff| {md:.2e}"
                         f"{'' if same_disk else '  (DISK TENSORS DIFFER)'}{info}")
            fails += int(not eq) + int(not same_disk)
            mx.clear_cache()
        del m_res, m_st1, m_org, full_c, full_s, w
        mx.clear_cache()
    kn = sorted({k.split("|")[0] for k in ns["_KERNELS"] if isinstance(k, str)})
    lines.append(f"reference kernels exercised: {kn}")
    txt = "\n".join(lines)
    print(txt)
    if outdir:
        pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
        (pathlib.Path(outdir) / f"partial_L{layers}.txt").write_text(txt + f"\nfails={fails}\n")
    return 1 if fails else 0


# ------------------------------------------------------------ selftest (CPU)
def _src_strings(path, names):
    """Kernel source strings of a runtime/model.py FILE, by AST (no import)."""
    sys.path.insert(0, str(HERE))
    import sz_resident
    return sz_resident.kernel_srcs_from_text(pathlib.Path(path).read_text(), tuple(names))


def selftest():
    sys.path.insert(0, str(HERE))
    import sz_resident
    import skipzero_load
    ok = True
    rng = np.random.default_rng(SEED)
    E, OUT, W, G = 6, 77, 5, 3
    live = rng.random((E, OUT)) > 0.7
    live[0] = False
    rowmask = np.packbits(live, axis=-1, bitorder="little")
    tbl = sz_resident.row_table_np(rowmask, E, OUT)
    comp = rng.integers(0, 2**32, (int(live.sum()), W), dtype=np.uint64).astype(np.uint32)
    full = skipzero_load.expand_np([E, OUT, W, G], rowmask, comp)
    # (1) the row table addresses exactly the rows the stage-1 expansion places
    g = np.where(tbl[..., None] >= 0, comp[np.maximum(tbl, 0)], 0).astype(np.uint32)
    t1 = g.tobytes() == full.tobytes() and (tbl < 0).sum() == (~live).sum()
    print(f"[selftest] row table vs stage-1 expansion (numpy): {'OK' if t1 else 'FAIL'}")
    ok &= t1
    # (2) mlx-CPU row table == numpy row table; resident_weights shape contract
    try:
        import mlx.core as mx
        with mx.stream(mx.cpu):
            tm = sz_resident.row_table_mx(mx.array(np.array([E, OUT, W, G], np.int32)),
                                          mx.array(rowmask))
            w = sz_resident.resident_weights({
                "a.sz_shape": mx.array(np.array([E, OUT, W, G], np.int32)),
                "a.sz_rowmask": mx.array(rowmask), "a.sz_codes": mx.array(comp),
                "a.sz_scales": mx.array(np.zeros((comp.shape[0], G), np.float16)),
                "b.weight": mx.array(np.ones(3, np.float16))})
            mx.eval(tm, *w.values())
            t2 = (np.array(tm) == tbl).all() and set(w) == {"a.codes", "a.vq_scales",
                                                            "a.row_table", "b.weight"} \
                and w["a.codes"].shape == comp.shape and w["a.row_table"].dtype == mx.int32
            t2 &= sz_resident.resident_weights(w) is w          # idempotent
        print(f"[selftest] mlx-CPU row table + resident_weights: {'OK' if t2 else 'FAIL'}")
        ok &= bool(t2)
    except ImportError:
        print("[selftest] mlx not importable: SKIPPED the mlx-CPU row-table check")
    # (3)/(4) kernel forks apply to each vintage's text (no import, no GPU);
    # the vintage is detected from the text and the OTHER vintage's patch set
    # must refuse it.
    # Since 2026-09-29 runtime/vq_switch.py carries the SZ switch NATIVELY
    # (`#if SZ` in the walk decode kernel and gemmseg2), so the walk fork now
    # targets the last runtime WITHOUT it -- the text every walk-vintage
    # bundle built before then carries (WALK_FROZEN).
    names = ["_SRC_FUSED_PACKED", "_SRC_FUSED_PACKED_D4_WALK", "_SRC_GEMMSEG2"]
    rt_txt = (SRC / "vqlab/runtime/vq_switch.py").read_text()
    native = (rt_txt.count("#if SZ") >= 2 and "def skipzero_weights" in rt_txt
              and "row_table=None" in rt_txt)
    print(f"[selftest] runtime/vq_switch.py carries the SZ switch natively: "
          f"{'OK' if native else 'FAIL'}")
    ok &= native
    for label, path, want in (("walk runtime (frozen)", WALK_FROZEN, "walk"),
                              ("397B bundle model.py", pathlib.Path(U8_MODEL_PY), "u8")):
        if not path.exists():
            if want == "u8":   # a local artifact, not package data
                print(f"[selftest] {label}: {path} not on this machine: SKIPPED")
                continue
            print(f"[selftest] {label}: {path} not found: FAIL")
            ok = False
            continue
        if want == "u8":
            txt = path.read_text()
            if txt.count("#if SZ") >= 2 and "def skipzero_weights" in txt:
                # rebundled onto a runtime with the SZ switch built in (the
                # 397B was, 2026-10-01): no fork applies, none is needed
                print(f"[selftest] {label}: carries the SZ switch natively (rebundled), "
                      "no u8 fork needed: OK")
                continue
        s = _src_strings(path, names)
        try:
            v, d, p = sz_resident.patch_vintage(s)
            t = (v == want and d.count("rowtbl") == 1 and p.count("rowtbl") == 1
                 and "sz_lr < 0" in d)
            if want == "u8":
                t &= "#if BITS != 0 || D_BAKE != 4" in p and "(size_t)e * OUT * WPR" not in p
                t &= "srow_w = scales\n" not in p
                try:           # the walk gemmseg patch must REFUSE this text
                    sz_resident.patch_gemmseg_src(s["_SRC_GEMMSEG2"])
                    t = False
                except RuntimeError:
                    pass
            else:
                try:
                    sz_resident.patch_gemmseg_u8_src(s["_SRC_GEMMSEG2"])
                    t = False
                except RuntimeError:
                    pass
        except (RuntimeError, NotImplementedError) as e:
            print(e)
            t, v = False, None
        print(f"[selftest] vintage '{v}' detected + forks apply to {label}: {'OK' if t else 'FAIL'}")
        ok &= bool(t)
    # (5) u8 compact rows through the row table == stage-1 expansion (numpy)
    comp8 = rng.integers(0, 256, (int(live.sum()), 8)).astype(np.uint8)
    full8 = skipzero_load.expand_np([E, OUT, 8, G], rowmask, comp8)
    g8 = np.where(tbl[..., None] >= 0, comp8[np.maximum(tbl, 0)], 0).astype(np.uint8)
    t5 = g8.tobytes() == full8.tobytes() and (comp8.view(np.uint32).view(np.uint8) == comp8).all()
    print(f"[selftest] u8 compact rows via row table vs stage-1 expansion: {'OK' if t5 else 'FAIL'}")
    ok &= bool(t5)
    print("[selftest] SKIPPED (need the GPU): kernel execution -- run "
          "`vqlab sz-bitexact --synthetic` and the model-mode gate")
    return 0 if ok else 1


# --------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(prog="vqlab sz-bitexact", description=__doc__.split("\n\n")[0])
    ap.add_argument("resident", nargs="?")
    ap.add_argument("reference", nargs="?")
    ap.add_argument("--ref2")
    ap.add_argument("--mem-ref")
    ap.add_argument("--out")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--partial", metavar="LAYERS",
                    help="real-weights module test for layers A-B: <resident> <stage1_pack> "
                         "[--original <rung>] (no full model load)")
    ap.add_argument("--original")
    ap.add_argument("--vintage", choices=("walk", "u8", "both", "native", "all"), default="both",
                    help="--synthetic: which runtime vintage(s) to stress")
    ap.add_argument("--u8-model-py", default=U8_MODEL_PY,
                    help="--synthetic: the u8-vintage bundle model.py (read only)")
    ap.add_argument("--worker", nargs=3, metavar=("ART", "ROLE", "EXPERTS_JSON"),
                    help=argparse.SUPPRESS)
    ap.add_argument("--mem-only", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.out:
        config.require_storage(a.out)
    if a.worker:
        return worker(a.worker[0], a.worker[1], a.out, a.worker[2], a.mem_only)
    if a.partial:
        if not (a.resident and a.reference):
            ap.error("--partial needs <resident> <stage1_pack>")
        return partial(a.resident, a.reference, a.partial, a.out, a.original)
    if a.synthetic:
        return synthetic(a.out, a.vintage, a.u8_model_py)
    if not (a.resident and a.reference and a.out):
        ap.error("need <resident> <reference> --out")
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ej = out / "experts.json"
    json.dump(dead_experts(a.resident), open(ej, "w"))
    jobs = [("resident", a.resident, False), ("reference", a.reference, False)]
    if a.ref2:
        jobs.append(("ref2", a.ref2, False))
    if a.mem_ref:
        jobs.append(("memref", a.mem_ref, True))
    env = dict(os.environ, PYTHONPATH=str(SRC))
    for role, art, mo in jobs:
        _log(f"worker {role}: {art}")
        cmd = [sys.executable, __file__, "--out", str(out), "--worker", art, role, str(ej)]
        if mo:
            cmd.append("--mem-only")
        r = subprocess.run(cmd, env=env)
        if r.returncode:
            _log(f"worker {role} FAILED rc={r.returncode}")
            return 2
    # The GATE is resident vs the STAGE-1 pack (--ref2): same live rows, dead
    # rows exact zero in both, so stage 2 must match it byte for byte. Against
    # the ORIGINAL, intermediate module outputs legitimately differ where the
    # original's dead rows held tiny non-zero values (stage 1 zeroed them) --
    # reported, not gated (35B, 2026-09-28: logits/prefill/greedy still equal).
    info = compare(out, "resident", "reference")
    fails = compare(out, "resident", "ref2") if a.ref2 else info
    mem_table(out, [j[0] for j in jobs])
    if a.ref2:
        print(f"\nvs original (information): {info} unequal checks -- stage 1's zeroed dead rows")
    print("\nVERDICT (vs " + ("stage-1 pack" if a.ref2 else "reference") + "):",
          "BYTE-EQUAL" if fails == 0 else f"{fails} UNEQUAL -- a bug to find")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
