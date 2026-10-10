"""Generate with MTP speculative drafting.

    python -m vqlab.cli mtp-generate --model <artifact> [--sidecar <file>]
        [--prompt ...] [--tokens N] [--temp 0.7] [--top-p 0.9]

A thin CLI over `vqlab.mtp.mtp_stream_generate`; the loop itself is library
code, so anything this script can do is three lines in a program:

    model, tok = mlx_lm.load(path, trust_remote_code=True)
    head, _ = vqlab.load_mtp_head(model, model_path=path)
    vqlab.mtp_generate(model, tok, "…", head, temp=0.7)

`vqlab mtp-bench` is the separate measurement entry point.
"""
import argparse
import sys


from vqlab.mtp import mtp_stream_generate
from vqlab.mtp.cli_common import add_sampling_args, load_all, prompt_ids  # noqa: F401  (re-exported)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sidecar", default=None,
                    help="quantized MTP sidecar (default: the family's "
                         "sidecar name in the artifact dir, if present)")
    ap.add_argument("--family", default=None,
                    help="override the family resolved from model_type")
    ap.add_argument("--prompt", default="Explain why vector quantization "
                    "compresses neural network weights better than scalar "
                    "rounding.")
    ap.add_argument("--tokens", type=int, default=128)
    add_sampling_args(ap)
    a = ap.parse_args()

    model, tok, head = load_all(a)
    last = None
    for r in mtp_stream_generate(model, tok, prompt_ids(tok, a.prompt), head,
                                 max_tokens=a.tokens, temp=a.temp,
                                 top_p=a.top_p, min_p=a.min_p, top_k=a.top_k,
                                 family=a.family, align=a.align):
        print(r.text, end="", flush=True)
        last = r
    if last is not None:
        print(f"\n\n{last.generation_tokens} tokens, "
              f"{last.generation_tps:.2f} tok/s, acceptance "
              f"{last.acceptance:.3f} over {last.steps} steps "
              f"({last.finish_reason})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
