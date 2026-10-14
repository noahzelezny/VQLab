"""Graft and sidecar plumbing shared by the per-family MTP heads and the
MTP probes: upstream `mtp.*` keys -> the trunk's layer layout, the
fail-loudly slot check, and the sidecar header read."""
from __future__ import annotations

import json

import mlx.core as mx
from mlx.utils import tree_flatten


def graft_layer_weights(g, experts=None):
    """`layers.0.*` graft tensors keyed the way the trunk's sanitize leaves
    its own layers: the fused `gate_up_proj` split in two, `experts` renamed
    `switch_mlp`. With an `experts` dict, any other `mlp.experts.<i>.<proj>`
    key (the unfused layout) lands in experts[proj][i] instead."""
    layer_w = {}
    for k, v in g.items():
        if not k.startswith("layers.0."):
            continue
        k = k[len("layers.0."):]
        if k.endswith("mlp.experts.gate_up_proj"):        # fused variant
            mid = v.shape[-2] // 2
            layer_w["mlp.switch_mlp.gate_proj.weight"] = v[..., :mid, :]
            layer_w["mlp.switch_mlp.up_proj.weight"] = v[..., mid:, :]
        elif k.endswith("mlp.experts.down_proj"):
            layer_w["mlp.switch_mlp.down_proj.weight"] = v
        elif experts is not None and k.startswith("mlp.experts."):  # unfused
            _, _, idx, proj, _ = k.split(".", 4)
            experts.setdefault(proj, {})[int(idx)] = v
        else:
            layer_w[k] = v
    return layer_w


def check_graft_slots(layer_w, module, *, note_missing=True):
    """FAIL on any graft key with no parameter slot in `module`; optionally
    note the slots the graft leaves at init."""
    slots = {k for k, _ in tree_flatten(module.parameters())}
    unmatched = sorted(set(layer_w) - slots)
    if unmatched:
        raise SystemExit(f"FAIL: {len(unmatched)} graft keys found no "
                         f"parameter slot, e.g. {unmatched[:4]}")
    if note_missing:
        missing = sorted(slots - set(layer_w))
        if missing:
            print(f"note: {len(missing)} module params not in graft (left "
                  f"init), e.g. {missing[:4]}", flush=True)


def read_sidecar(path):
    """(tensors, vqlab_mtp config, bits, group_size) of a head sidecar."""
    w = mx.load(str(path))
    meta = mx.load(str(path), return_metadata=True)[1]
    cfg = json.loads(meta.get("vqlab_mtp", "{}"))
    return w, cfg, cfg.get("bits"), cfg.get("group_size", 32)
