"""CLI plumbing shared by the MTP tools: `vqlab mtp-generate` (ship/mtp_run)
and `vqlab mtp-bench` (bench/mtp_bench). Sampling flags, loading trunk +
drafting head, the chat-templated prompt. In the library so a bench does not
import a ship tool.
"""
import mlx.core as mx

from vqlab.mtp import load_mtp_head
from vqlab.util.fs import GIB


def add_sampling_args(ap):
    ap.add_argument("--temp", type=float, default=0.0,
                    help="0 (default) is greedy; above 0 uses exact rejection "
                         "sampling, which preserves the output distribution")
    ap.add_argument("--top-p", type=float, default=0.0)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--align", default="committed",
                    choices=["committed", "legacy"],
                    help="head-cache scheme; 'legacy' is the pre-2026-08-31 "
                         "misaligned loop, kept only for A/B measurement")
    return ap


def load_all(a):
    from vqlab.family.model_loader import load_artifact
    model, tok = load_artifact(a.model)
    before = mx.get_active_memory()
    head, spec = load_mtp_head(model, sidecar=a.sidecar, model_path=a.model,
                               family=a.family)
    mx.clear_cache()
    print(f"MTP head ({spec.name}) resident: "
          f"{(mx.get_active_memory() - before) / GIB:.2f} GiB", flush=True)
    return model, tok, head


def prompt_ids(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}],
                                   add_generation_prompt=True)
