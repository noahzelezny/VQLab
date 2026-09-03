"""Per-token weight bytes of an MoE artifact, from safetensors HEADERS only.

A decode step reads ALL the dense weights but only top_k of n_experts expert
tensors, so an MoE artifact's TOTAL size is a poor predictor of its decode
cost.  This computes the number that does predict it.

Written for the 2026-09-02 dispatch arc, to test whether the cluster result
"VQ 2.6bpw and 3.1bpw decode at the SAME speed despite 21 GiB of size
difference" needs a latency floor to explain it.  It does not: their
per-token reads are within 3.7%.  See the ledger entry
"2026-09-02 -- DISPATCH-REDUCTION ARC".

Reads headers only -- no tensor data, no model load, a few hundred KB of IO.

Run:  python scripts/per_token_bytes.py
"""
import json, os, struct, sys

ROOT = "/Volumes/Thunderbay SSD/Exo Models"
DTS = {"F32": 4, "F16": 2, "BF16": 2, "U8": 1, "I8": 1, "U16": 2, "U32": 4,
       "I32": 4, "I64": 8, "F8_E4M3": 1, "F8_E5M2": 1}


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def scan(name, topk, n_exp):
    d = os.path.join(ROOT, name)
    shards = sorted(x for x in os.listdir(d) if x.endswith(".safetensors"))
    dense = experts = 0
    for s in shards:
        for k, v in header(os.path.join(d, s)).items():
            if k == "__metadata__":
                continue
            nb = 1
            for x in v["shape"]:
                nb *= x
            nb *= DTS[v["dtype"]]
            # expert-axis tensors: leading dim == n_experts
            if v["shape"] and v["shape"][0] == n_exp and (
                    "experts" in k or "switch_mlp" in k or "mlp" in k):
                experts += nb
            else:
                dense += nb
    per_tok = dense + experts * topk / n_exp
    print(f"{name}")
    print(f"   dense           {dense/2**30:8.3f} GiB")
    print(f"   experts (all)   {experts/2**30:8.3f} GiB")
    print(f"   experts/token   {experts*topk/n_exp/2**30:8.3f} GiB "
          f"(top{topk} of {n_exp})")
    print(f"   PER-TOKEN READ  {per_tok/2**30:8.3f} GiB")
    print(f"   TOTAL           {(dense+experts)/2**30:8.3f} GiB\n")
    return per_tok


a = scan("TheDrainFlorist--Qwen3.5-397B-A17B-VQ-2.6bpw", 8, 512)
b = scan("TheDrainFlorist--Qwen3.5-397B-A17B-VQ-3.1bpw", 8, 512)
c = scan("spicyneuron--Qwen3.5-397B-A17B-MLX-2.6bit", 8, 512)
print(f"per-token ratio  VQ3.1 / VQ2.6 = {b/a:.3f}")
print(f"per-token ratio  affine2.6 / VQ2.6 = {c/a:.3f}")
