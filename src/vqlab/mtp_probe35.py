"""MTP acceptance probe for qwen3_5 / qwen3_5_moe, sweeping the wiring
choices the checkpoint does not determine.

    python -m vqlab.cli mtp-probe35 --model <artifact> --sidecar <s.safetensors>
        --corpus <txt> [--tokens 512] [--out probe.json]

Teacher-forced greedy proxy for speculative acceptance: run the MAIN model
over the corpus once, capturing the hidden row h_i that feeds its final norm
and the main model's own greedy prediction at every position. Then run the
head at each position i with (h_i, embedding of the ACTUAL next token) and ask
how often its greedy draft for token i+2 matches what the main model itself
would produce there. That match rate upper-bounds greedy speculative
acceptance; ~chance means the head, or this wiring, is dead.

Why a sweep and not a guess. Three things about this family's head are not
recoverable from the checkpoint -- the RMSNorm delta convention, the concat
order into the fused `fc`, and whether the head reads the trunk hidden state
before or after the trunk's final norm. Each wrong choice produces near-zero
acceptance with no error, which is exactly the failure that made the qwen4_exp
head look dead for a day. `norm_shift` is fixed at pack time (it changes the
stored weights, so it is one sidecar per value); the other two are pure
inference-time wiring, so one sidecar load sweeps both.

The instrument is cheap: ONE trunk forward, then one head forward per wiring.
"""
import argparse
import json
import pathlib
import sys

import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from vqlab.mtp import registry
from vqlab.mtp.capture import capture_input


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="main-model artifact dir")
    ap.add_argument("--sidecar", required=True, nargs="+",
                    help="one or more packed sidecars; all are scored\n"
                         "against a single model load")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--family", default=None)
    ap.add_argument("--fc-orders", default="eh,he")
    ap.add_argument("--h-sources", default="pre_norm,post_norm")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from mlx_lm.utils import load
    model, tok = load(a.model, lazy=False, trust_remote_code=True)
    spec = registry.resolve(model, a.family)
    arch = spec.arch_module(model)
    print(f"family {spec.name}; capture {spec.capture!r}; "
          f"resident {mx.get_active_memory() / 2**30:.2f} GiB", flush=True)

    ids = tok.encode(open(a.corpus).read())[: a.tokens + 2]
    bos = getattr(tok, "bos_token_id", None)
    if bos is not None and (not ids or ids[0] != bos):
        ids = [bos] + ids[: a.tokens + 1]
    S = len(ids) - 2                      # positions with an i+2 target
    inp = mx.array([ids[:-1]])            # main model sees tokens 0..S
    nxt = mx.array([ids[1:]])             # token t+1 at every position

    # ---- one trunk pass: hidden row + the trunk's own greedy choice ------
    with capture_input(model.model, spec.capture) as get_h:
        logits = model(inp)
        mx.eval(logits)
        h = get_h()
    mx.eval(h)
    main_pred = mx.argmax(logits.astype(mx.float32), axis=-1)[0]   # [S+1]
    mx.eval(main_pred)
    del logits
    mx.clear_cache()
    ctrl = float(mx.mean((main_pred[:S] == mx.array(ids[1:S + 1])
                          ).astype(mx.float32)).item())
    print(f"control: main greedy vs corpus next-token agreement {ctrl:.3f} "
          f"(sanity: ~0.4-0.7; ~0 means the capture is broken)", flush=True)
    if ctrl < 0.15:
        raise SystemExit("FAIL: the control says the trunk capture is wrong; "
                         "no acceptance number from this run means anything")

    rec = {"model": a.model, "corpus": a.corpus, "positions": S,
           "family": spec.name, "control_main_vs_corpus": round(ctrl, 4),
           "sidecars": {}}
    orders = [s for s in a.fc_orders.split(",") if s]
    sources = [s for s in a.h_sources.split(",") if s]

    for path in a.sidecar:
        name = pathlib.Path(path).name
        print(f"=== {name} ===", flush=True)
        head = spec.head_cls().from_sidecar(model, arch, path)
        res = {}
        for fc_order in orders:
            for h_source in sources:
                head.fc_order, head.h_source = fc_order, h_source
                lg = head.draft_logits(h, nxt, None).astype(mx.float32)
                pred = mx.argmax(lg, axis=-1)[0]
                mx.eval(pred)
                del lg
                mx.clear_cache()
                # The head at position i drafts token i+2; the main model's
                # own greedy choice for i+2 is main_pred[i+1].
                hits = (pred[:S] == main_pred[1:S + 1])
                acc = float(mx.mean(hits.astype(mx.float32)).item())
                accc = float(mx.mean((pred[:S] == mx.array(ids[2:S + 2]))
                                     .astype(mx.float32)).item())
                key = f"{fc_order}/{h_source}"
                res[key] = {"acceptance_vs_main_greedy": round(acc, 4),
                            "agreement_vs_corpus": round(accc, 4),
                            "hits": int(mx.sum(hits).item())}
                sample = tok.decode([int(x) for x in pred[:8].tolist()])
                print(f"  {key:22s} acc_vs_main {acc:.4f} "
                      f"({int(mx.sum(hits).item())}/{S})  "
                      f"vs_corpus {accc:.4f}  sample: {sample!r}", flush=True)
        best = max(res, key=lambda k: res[k]["acceptance_vs_main_greedy"])
        print(f"  best: {best} at "
              f"{res[best]['acceptance_vs_main_greedy']:.4f}", flush=True)
        rec["sidecars"][name] = {"wirings": res, "best": best}
        del head
        mx.clear_cache()

    print(json.dumps(rec), flush=True)
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
