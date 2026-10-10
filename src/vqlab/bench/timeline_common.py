"""What the timeline benches (decode-timeline, prefill-timeline) share: how
to find a model's decoder trunk and how to classify one decoder layer."""
from __future__ import annotations


def trunk(m, depth=0):
    """The module carrying both `layers` and `embed_tokens`. Requiring BOTH
    stops the walk at the right level (a wrapper can have `layers` without
    `embed_tokens`, which is how the first version of this landed on the
    wrong module). None if no such module within 6 wrappers."""
    if depth > 6:
        return None
    if hasattr(m, "layers") and hasattr(m, "embed_tokens"):
        return m
    for attr in ("language_model", "model"):
        sub = getattr(m, attr, None)
        if sub is not None and sub is not m:
            got = trunk(sub, depth + 1)
            if got is not None:
                return got
    return None


def layer_kind(layer):
    """(attention-half kind, MLP-half kind) of one decoder layer:
    ("lin" | "attn", "moe" | "dense")."""
    attn = "lin" if getattr(layer, "is_linear", False) else "attn"
    mlp = getattr(layer, "mlp", None)
    moe = mlp is not None and any(hasattr(mlp, a) for a in
                                  ("switch_mlp", "experts", "gate"))
    return attn, ("moe" if moe else "dense")
