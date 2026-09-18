"""vqlab prefill-bench — time PREFILL on one artifact under both codebook arms.

WHY PREFILL. I.9: decode is a wash across geometries; prefill is where
geometry shows, and `gemmseg` -- the fused segmented VQ-GEMM -- is the
prefill kernel whose codebook arm is in question.

WHY ONE ARTIFACT. The arm is chosen by `vq_switch.gemmseg_cb_dev(D, K)`:

    cb + tiles + xtpad > _TG_CAP_BYTES  OR  cb >= 16384

d4-K2048's codebook is 16384 B and tiles are 12288, so 28672 <= 32768 -- it
PHYSICALLY FITS threadgroup and is sent to device purely by the second
clause, a preference added 2026-09-08. So the same weights can run both
arms, which makes this a clean A/B with no refit, no rebuild and no
confound: identical bytes, identical numerics (that is what e134_accept
verifies), only the arm differs.

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


def _arm_of(vq, D, K):
    return "device" if vq.gemmseg_cb_dev(D, K) else "threadgroup"


def _force(vq, mode):
    """Force every geometry onto one arm. Returns a restore callable."""
    orig = vq.gemmseg_cb_dev
    if mode == "device":
        vq.gemmseg_cb_dev = lambda D, K: True
    elif mode == "threadgroup":
        # Only lift the PREFERENCE, never the hardware cap: a codebook that
        # does not fit must still take the device arm or the kernel would
        # over-allocate threadgroup memory, which Metal reports as
        # XPC_ERROR_CONNECTION_INTERRUPTED rather than a clean error (IV).
        def tg(D, K):
            cb = K * 2 * D
            return cb + vq._TILES_R32 + vq._XT_PAD_BYTES_R32 > vq._TG_CAP_BYTES
        vq.gemmseg_cb_dev = tg
    else:
        raise SystemExit(f"FAIL: unknown arm {mode!r}")
    return lambda: setattr(vq, "gemmseg_cb_dev", orig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--arms", default="device,threadgroup")
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
    print(f"model_type {mt}; geometries and their DEFAULT arm:")
    for D, K in geos:
        print(f"  d{D}-K{K:<6d} cb {K*2*D:7d} B -> {_arm_of(vq, D, K)}")

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
    for arm in arms:
        restore = _force(vq, arm)
        try:
            one_pass()                                   # warm-up, discarded
            ts = []
            for _ in range(a.reps):
                mx.clear_cache()
                t0 = time.perf_counter()
                one_pass()
                ts.append(time.perf_counter() - t0)
            res[arm] = ts
            print(f"  {arm:12s} median {statistics.median(ts):7.3f}s  "
                  f"min {min(ts):7.3f}s  spread "
                  f"{(max(ts)-min(ts))/statistics.median(ts)*100:4.1f}%",
                  flush=True)
        finally:
            restore()

    if len(res) == 2:
        d, t = statistics.median(res["device"]), statistics.median(res["threadgroup"])
        print(f"\nRATIO threadgroup/device = {t/d:.3f}  "
              f"({'threadgroup faster' if t < d else 'device faster'} "
              f"by {abs(1 - t/d)*100:.1f}%)")
        print("Quote the RATIO, not the absolutes (quantlab III).")
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(
            {"model": str(mp), "tokens": int(ids.shape[1]), "reps": a.reps,
             "times": res}, indent=1))


if __name__ == "__main__":
    main()
