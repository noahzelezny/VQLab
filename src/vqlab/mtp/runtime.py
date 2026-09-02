"""Trunk loading for the MTP tools, across BOTH runtimes.

mlx-lm serves most families; glm5_next exists only in mlx_vlm (which is
also exo's serving path for it). One loader keeps every mtp-* entry
point runtime-agnostic: it returns (model-to-bind, tokenizer), where the
model is the object the registry contract expects — mlx-lm's Model, or
the mlx_vlm LanguageModel (never the VLM wrapper).
"""
from __future__ import annotations

import importlib
import json
import pathlib


def load_trunk(model_path, lazy: bool = False):
    model_path = pathlib.Path(model_path)
    model_type = json.load(open(model_path / "config.json")).get("model_type")
    try:
        importlib.import_module(f"mlx_lm.models.{model_type}")
        have_mlx_lm = True
    except ImportError:
        have_mlx_lm = False
    if have_mlx_lm:
        from mlx_lm.utils import load
        try:
            return load(model_path, lazy=lazy, trust_remote_code=True)
        except TypeError:  # older mlx-lm: no trust_remote_code kwarg
            return load(model_path, lazy=lazy)
    from mlx_vlm.utils import load as vlm_load
    model, processor = vlm_load(str(model_path), lazy=lazy)
    tok = getattr(processor, "tokenizer", processor)
    return getattr(model, "language_model", model), tok


def encode_chat(tok, text):
    """Chat-template a single user message to token ids, whichever of the
    three shapes this tokenizer's apply_chat_template returns (ids, a
    rendered string, or a list of rendered strings — bare HF tokenizers
    from mlx_vlm do the latter two)."""
    ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                  add_generation_prompt=True)
    if isinstance(ids, str):
        return tok.encode(ids)
    if ids and isinstance(ids[0], str):
        return tok.encode("".join(ids))
    return ids
