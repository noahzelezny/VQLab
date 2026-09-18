"""vqlab prefill-bench — time PREFILL on one artifact under both codebook arms.

WHY PREFILL. I.9: decode is a wash across geometries; prefill is where
geometry shows, and `gemmseg` -- the fused segmented VQ-GEMM -- is the
prefill kernel whose codebook arm is in question.

WHY ONE ARTIFACT. d4-K2048's codebook is 16384 B against a 12288 B tile
budget under a 32768 B cap, so 28672 <= 32768: BOTH ARMS ARE LEGAL for it.
`VQ_MOE_GEMMSEG_CBDEV` forces either ("1" device, "0" threadgroup, "auto"
by budget), so the same weights run both arms -- identical bytes, identical
numerics (what e134_accept verifies), only the arm differs. vq_switch's own
comment says nobody had measured which is faster.

DO NOT USE `gemmseg_cb_dev` TO SELECT THE ARM. It is a REPORTING predicate
for `coverage`, it carries an extra `or cb >= 16384` clause the real
selection does not, and nothing on the prefill path calls it. A first
version of this bench monkeypatched it after load and measured the same
code path twice, reporting a 0.997 ratio that meant nothing.

THE ENV VAR IS READ AT IMPORT, so each arm needs its OWN PROCESS -- which
the speed rules require anyway (one process per arm).

SPEED RULES (quantlab III), because this instrument has burned people:
  * n >= 3 per arm, alternating, ONE process, same session.
  * Report the RATIO between arms, never an absolute -- at ~100 GiB the
    decode instrument is BIMODAL (21.1 / 12.7 / 21.3 / 21.2 tok/s on one
    artifact back to back, cause unknown).
  * A discarded warm-up per arm: the first call pays kernel compilation.
  * State the prompt length.

    vqlab prefill-bench --model <dir> --tokens 2048 --reps 5
"""
import argparse
import importlib
import json
import pathlib
import statistics
import sys
import time

import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import runtime_load


ARM_ENV = {"device": "1", "threadgroup": "0", "auto": "auto"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--arms", default="device,threadgroup")
    ap.add_argument("--_child", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--drop-first-child", action="store_true", default=True,
                    help="discard the FIRST child's timings entirely. The "
                         "session's first model load runs on a cold machine "
                         "and produced a 163%% spread with a 6.7s median "
                         "against 2.7s warm (2026-09-17); a discarded rep is "
                         "not enough, the whole first process is the outlier.")
    ap.add_argument("--rounds", type=int, default=1,
                    help="alternate the arms this many times (A,B,B,A...). "
                         "ORDER IS A CONFOUND: whichever arm runs first pays "
                         "a colder machine, and on 2026-09-17 the device arm "
                         "went first and returned a 97.3%% spread whose "
                         "median and min disagreed in DIRECTION.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    mp = pathlib.Path(a.model)
    cfg = json.load(open(mp / "config.json"))
    mt = cfg.get("model_type") or cfg.get("text_config", {}).get("model_type")
    fam = {"qwen4_exp": "qwen4_exp", "glm5_next": "glm5_next"}.get(mt, mt)

    import vq_switch as vq
    geos = sorted({(v.get("dim"), v.get("k"))
                   for v in (cfg.get("vq_modules") or {}).values()
                   if v.get("dim") and v.get("k")})
    print(f"model_type {mt}; geometries (cb bytes vs the 32768 B cap, of "
          f"which {vq._TILES_R32} B is tiles):")
    for D, K in geos:
        cb = K * 2 * D
        legal = cb + vq._TILES_R32 + vq._XT_PAD_BYTES_R32 <= vq._TG_CAP_BYTES
        print(f"  d{D}-K{K:<6d} cb {cb:7d} B  threadgroup-legal={legal}"
              f"{'  <- both arms legal, this is the A/B' if legal else ''}")

    if a._child:
        import os
        import vq_switch as vq
        # PROVE THE ARM CHANGED. Twice today a tool reported success while
        # doing nothing (ple_stream's instance patch; this bench's first
        # version patching a REPORTING predicate). The device arm builds
        # kernels named *_devcb, so record what actually gets compiled and
        # print it -- a bench that cannot show the two arms ran different
        # kernels is not measuring an arm.
        built = []
        _mk = mx.fast.metal_kernel
        def _rec(*args, **kw):
            built.append(kw.get("name") or (args[0] if args else "?"))
            return _mk(*args, **kw)
        mx.fast.metal_kernel = _rec
        model, _ = runtime_load.load_for_family(fam, mp, lazy=False)
        from mlx_lm.utils import load_tokenizer as _lt
        tok = _lt(mp)
        text = (open(a.corpus).read() if a.corpus
                else "The quick brown fox jumps over the lazy dog. " * 4000)
        ids = mx.array([tok.encode(text)[: a.tokens]])
        mx.eval(model(ids))                       # warm-up, discarded
        ts = []
        for _ in range(a.reps):
            mx.clear_cache()
            t0 = time.perf_counter(); mx.eval(model(ids))
            ts.append(time.perf_counter() - t0)
        devcb = sorted({n for n in built if "devcb" in str(n)})
        tg = sorted({n for n in built if "gemmseg" in str(n)
                     and "devcb" not in str(n)})
        print(f"KERNELS {a._child} cbdev_env={vq._GEMMSEG_CBDEV} "
              f"devcb={len(devcb)} other_gemmseg={len(tg)} "
              f"sample={(devcb or tg)[:2]}", flush=True)
        print("TIMES " + a._child + " " + " ".join(f"{t:.6f}" for t in ts),
              flush=True)
        return

    model, _ = runtime_load.load_for_family(fam, mp, lazy=False)
    from mlx_lm.utils import load_tokenizer
    tok = load_tokenizer(mp)
    text = (open(a.corpus).read() if a.corpus
            else "The quick brown fox jumps over the lazy dog. " * 4000)
    ids = mx.array([tok.encode(text)[: a.tokens]])
    print(f"\nprompt {ids.shape[1]} tokens, {a.reps} reps/arm "
          f"(+1 discarded warm-up), alternating, one process", flush=True)

    def one_pass():
        mx.eval(model(ids))

    res = {}
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    order = []
    for r in range(a.rounds):
        order += arms if r % 2 == 0 else arms[::-1]
    import os, subprocess
    for arm in order:
        env = dict(os.environ, VQ_MOE_GEMMSEG_CBDEV=ARM_ENV[arm])
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--model", a.model,
             "--tokens", str(a.tokens), "--reps", str(a.reps),
             "--_child", arm] + (["--corpus", a.corpus] if a.corpus else []),
            capture_output=True, text=True, env=env)
        for k in (l for l in r.stdout.splitlines() if l.startswith("KERNELS")):
            print("   " + k, flush=True)
        line = [l for l in r.stdout.splitlines() if l.startswith("TIMES ")]
        if not line:
            tail = (r.stderr or r.stdout).strip().splitlines()
            raise SystemExit(f"FAIL: {arm} arm produced no timings "
                             f"(rc={r.returncode}): "
                             + (tail[-1] if tail else "no output"))
        ts = [float(x) for x in line[-1].split()[2:]]
        if a.drop_first_child and not res:
            print(f"   (first child discarded: cold machine)", flush=True)
            res.setdefault(arm, [])
            continue
        res.setdefault(arm, []).extend(ts)
        print(f"  {arm:12s} (VQ_MOE_GEMMSEG_CBDEV={ARM_ENV[arm]}) "
              f"median {statistics.median(ts):7.3f}s  min {min(ts):7.3f}s  "
              f"spread {(max(ts)-min(ts))/statistics.median(ts)*100:4.1f}%",
              flush=True)

    if len(res) == 2:
        d, t = res["device"], res["threadgroup"]
        for lbl, f in (("median", statistics.median), ("min", min)):
            rd, rt = f(d), f(t)
            print(f"\nRATIO by {lbl}: threadgroup/device = {rt/rd:.3f}  "
                  f"({'threadgroup' if rt < rd else 'device'} faster by "
                  f"{abs(1 - rt/rd)*100:.1f}%)   [{lbl} {rd:.3f} vs {rt:.3f}]")
        sd = (max(d) - min(d)) / statistics.median(d) * 100
        st = (max(t) - min(t)) / statistics.median(t) * 100
        if sd > 25 or st > 25 or (statistics.median(t) < statistics.median(d))\
                != (min(t) < min(d)):
            print("\n!! DO NOT QUOTE THIS RATIO. Spread "
                  f"{sd:.0f}%/{st:.0f}%, or median and min disagree in "
                  "DIRECTION -- the instrument is bimodal here (quantlab III) "
                  "and neither statistic is describing the kernel.")
        else:
            print("Quote the RATIO, not the absolutes (quantlab III).")
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(
            {"model": str(mp), "tokens": int(ids.shape[1]), "reps": a.reps,
             "times": res}, indent=1))


if __name__ == "__main__":
    main()
