"""Emit an MTP head in the layout a native qwen4_exp runtime expects.

    python -m vqlab.cli mtp-graft --model <artifact> --mtp <graft.safetensors>
        [--bits 8] [--expert-bits 4] [--out <dir>] [--reference <index.json>]

`vqlab mtp-pack` writes VQLab's OWN sidecar: internal key names, a fused fc,
and a filename deliberately outside mlx-lm's `model*.safetensors` glob so the
head costs nothing until something asks for it. That file is for
`vqlab mtp-generate`, and nothing else can read it (its metadata says so:
`mtplx_compatible: "false"`).

This command writes the other thing — the same weights in the layout a runtime
with native qwen4_exp MTP looks for, verified key-for-key against a checkpoint
known to work with one:

  keys      `language_model.mtp.*`, matching the module tree exactly:
            block -> layers.0, mixer -> hyper_connection_mixer,
            norm_e -> pre_fc_norm_embedding, norm_h -> pre_fc_norm_hidden.
  fc        split back into fc_embedding / fc_hidden. VQLab fuses them into
            one matmul for its own loop; upstream keeps them separate and the
            runtime's own module expects that.
  filename  `model-mtp.safetensors`, i.e. INSIDE the glob — the opposite
            choice from mtp-pack, and deliberate. A native loader discovers
            weights by that glob and does not consult the index
            (jundot/omlx#2062), so a head it is supposed to find must match it.
  config    a `quantization` fragment naming every emitted module and the bits
            it actually got, since the recipe is mixed.

VERIFIED against Vontra/Qwen3.8-Flash-Next-MLX-oQ4-MTP (2026-08-31): 76
tensors, 32 modules, `mlp.gate` left raw. Our head's module tree is identical
to it, which is independent confirmation of the wiring in mtp_head.py.

NOT verified: that any runtime will load this INSIDE a VQLab VQ artifact. Our
artifacts carry `model_file: model.py` and need it for the VQ kernels; the
reference checkpoint has no `model_file` and uses the runtime's own class. See
the README for that open question — this command emits the weights, it does
not prove a runtime consumes them.
"""
import argparse
import importlib
import json
import pathlib
import sys

import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from mtp_head import MTPHead

PREFIX = "language_model.mtp."
OUT_NAME = "model-mtp.safetensors"

# our flat sidecar name -> upstream name, longest prefix first
_REMAP = (
    ("block.", "layers.0."),
    ("mixer.", "hyper_connection_mixer."),
    ("norm_e.weight", "pre_fc_norm_embedding.weight"),
    ("norm_h.weight", "pre_fc_norm_hidden.weight"),
)


def _rename(key):
    for src, dst in _REMAP:
        if key.startswith(src):
            return dst + key[len(src):] if src.endswith(".") else dst
    return key


def emit(head, bits, expert_bits, group_size):
    """Flat {upstream key: array} for the head, plus the quantization map."""
    from mlx.utils import tree_flatten
    flat, quant = {}, {}

    def add(name, arr):
        flat[PREFIX + name] = arr

    for owner, params in (("block", head.block), ("mixer", head.mixer)):
        for k, v in tree_flatten(params.parameters()):
            add(_rename(f"{owner}.{k}"), v)
    add("pre_fc_norm_embedding.weight", head.norm_e.weight.astype(mx.bfloat16))
    add("pre_fc_norm_hidden.weight", head.norm_h.weight.astype(mx.bfloat16))

    # Un-fuse fc: VQLab concatenates [fc_embedding | fc_hidden] along the
    # input axis for one matmul; the runtime's module wants them separate.
    D = head.D
    for name, half in (("fc_embedding", head.fc[:, :D]),
                       ("fc_hidden", head.fc[:, D:])):
        w, s, b = mx.quantize(half.astype(mx.bfloat16), group_size=group_size,
                              bits=bits)
        add(f"{name}.weight", w)
        add(f"{name}.scales", s)
        add(f"{name}.biases", b)

    # Every quantized module, with the bits it actually received.
    for key in flat:
        if key.endswith(".scales"):
            mod = key[len(PREFIX):-len(".scales")]
            quant[PREFIX + mod] = {
                "bits": expert_bits if "switch_mlp" in mod else bits,
                "group_size": group_size, "mode": "affine"}
    return flat, quant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mtp", required=True, help="bf16 mtp graft safetensors")
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--expert-bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--out", default=None, help="directory (default: cwd)")
    ap.add_argument("--reference", default=None,
                    help="JSON list of expected key names, or an "
                         "index.json — key-set parity is GATED on it")
    a = ap.parse_args()

    from mlx_lm.utils import load
    model, _ = load(a.model, lazy=True, trust_remote_code=True)
    arch = importlib.import_module(type(model.model).__module__)

    g = {k[len("mtp."):] if k.startswith("mtp.") else k: v
         for k, v in mx.load(a.mtp).items()}
    head = MTPHead(model, arch).load_graft(g)
    del g
    head.quantize(bits=a.bits, group_size=a.group_size,
                  expert_bits=a.expert_bits)
    flat, quant = emit(head, a.bits, a.expert_bits, a.group_size)

    out = pathlib.Path(a.out or ".")
    out.mkdir(parents=True, exist_ok=True)
    path = out / OUT_NAME
    mx.save_safetensors(str(path), flat, metadata={"format": "mlx"})
    cfg = out / "mtp-quantization.json"
    cfg.write_text(json.dumps(quant, indent=1))

    print(f"wrote {path}")
    print(f"  {len(flat)} tensors, {path.stat().st_size / 2**30:.2f} GiB, "
          f"experts {a.expert_bits}-bit / rest {a.bits}-bit / group "
          f"{a.group_size}")
    print(f"  matches the model*.safetensors glob: "
          f"{path.name.startswith('model')}")
    print(f"wrote {cfg}  ({len(quant)} quantized modules; merge into "
          f"config.json's `quantization`)")

    if a.reference:
        ref = json.loads(pathlib.Path(a.reference).read_text())
        if isinstance(ref, dict):
            ref = list(ref.get("weight_map", ref))
        want = {k for k in ref if ".mtp." in k}
        got = set(flat)
        missing, extra = sorted(want - got), sorted(got - want)
        print(f"\nkey-set parity vs reference: {len(got)} emitted, "
              f"{len(want)} expected")
        if missing or extra:
            for k in missing[:6]:
                print(f"  MISSING {k}")
            for k in extra[:6]:
                print(f"  EXTRA   {k}")
            raise SystemExit(f"FAIL: {len(missing)} missing, {len(extra)} "
                             f"extra — this will not load")
        print("  PASS: identical key sets")


if __name__ == "__main__":
    main()
