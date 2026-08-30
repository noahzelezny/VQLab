"""Pack a bf16 qwen4_exp MTP graft into a quantized drafting sidecar.

    python -m vqlab.cli mtp-pack --model <artifact> --mtp <graft.safetensors>
        [--bits 6] [--group-size 32] [--out <dir-or-file>]

The output is deliberately NOT named `model*.safetensors`. mlx-lm discovers
weights by globbing that pattern (utils.py:349) and never consults the index,
so a sidecar named `mtp-head-q6.safetensors` is invisible to the stock loader:
dropping it into an artifact directory costs nothing until something asks for
it by name. That is what makes the head optional.

Measured on Flash-Next 2.1bpw (2026-08-30): the 6-bit head is 2.12 GiB
resident and lifts greedy decoding from 16.07 to 26.87 tok/s (1.67x) at 0.708
draft acceptance. Head precision cannot affect output quality -- the trunk
verifies every drafted token, so a worse draft costs a rejection, never a
wrong token; 6-bit and bf16 measured identical acceptance.
"""
import argparse
import importlib
import pathlib
import sys

import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from mtp_head import SIDECAR_NAME, MTPHead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="artifact dir (loaded lazily; "
                    "no trunk weights are materialized)")
    ap.add_argument("--mtp", required=True, help="bf16 mtp graft safetensors")
    ap.add_argument("--bits", type=int, default=6)
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--out", default=None,
                    help="output file, or a directory to write "
                         f"{SIDECAR_NAME} into (default: the artifact dir)")
    a = ap.parse_args()

    from mlx_lm.utils import load
    # lazy=True: we need the architecture classes and args, not the weights.
    model, _ = load(a.model, lazy=True, trust_remote_code=True)
    arch = importlib.import_module(type(model.model).__module__)

    g = {k[len("mtp."):] if k.startswith("mtp.") else k: v
         for k, v in mx.load(a.mtp).items()}
    head = MTPHead(model, arch).load_graft(g)
    del g

    before = mx.get_active_memory()
    head.quantize(bits=a.bits, group_size=a.group_size)
    mx.clear_cache()
    resident = (mx.get_active_memory() - before) / 2**30

    out = pathlib.Path(a.out) if a.out else pathlib.Path(a.model)
    if out.is_dir():
        out = out / (SIDECAR_NAME if a.bits == 6
                     else f"mtp-head-q{a.bits}.safetensors")
    flat = head.save(out, bits=a.bits, group_size=a.group_size)
    size = out.stat().st_size / 2**30
    print(f"wrote {out}")
    print(f"  {len(flat)} tensors, {size:.2f} GiB on disk, "
          f"{a.bits}-bit / group {a.group_size}")
    print(f"  quantization changed resident by {resident:+.2f} GiB")
    print(f"  invisible to mlx-lm's model*.safetensors glob: "
          f"{not out.name.startswith('model')}")


if __name__ == "__main__":
    main()
