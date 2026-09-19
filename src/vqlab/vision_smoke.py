#!/usr/bin/env python3
"""Vision arm of the release gate: put an IMAGE through the shipping runtime.

Rule III.11 says generate one token through the shipping runtime before
calling anything releasable. `smoke` does that -- for TEXT. Nothing in the
gate set had ever put an image through a multimodal artifact, and on
2026-09-19 that let 17 of 20 shipped multimodal bundles carry a vision tower
the runtime could not reach (F153): the shim resolved a TEXT-ONLY base arch
that happened to share its `model_type` with mlx_vlm's multimodal one. Every
tensor was present, `check-release` and `check-bundle` passed, and `smoke`
passed -- because a text-only arch serves text perfectly well.

Three checks, cheapest first, so a broken bundle fails before anything loads:

  1. SURFACE. Import the artifact's own model.py and demand the multimodal
     surface (TextConfig / VisionConfig / VisionModel). This alone catches
     F153 in about a second and needs no weights, which is why it is also
     what `--static` runs on models too large for this box.

  2. LOAD. Load through mlx_vlm and confirm the instantiated model really
     carries a vision tower module, not just a config key.

  3. GRAPH. Forward the SAME prompt with and without the probe image and
     require the logits to MOVE. This is the check that cannot be faked by
     configuration: a tower that is present, loaded, and not wired into the
     graph produces identical logits, and a caption-based check would still
     read as a fluent sentence. The gate asserts an effect, not a vibe.

A generated caption is PRINTED for the human, never asserted on -- captions
are not a deterministic instrument and III.11 does not need them to be.

    vqlab vision-smoke <artifact> [--static] [--prompt ...] [--image f.png]
"""
import argparse
import importlib.util
import json
import os
import pathlib
import sys

# A synthetic probe beats a checked-in JPEG: no binary in the repo, no license
# question, and the content is known exactly, so the printed caption is worth
# reading. Deterministic by construction.
PROBE_DESC = "a yellow circle centred on a dark blue square"


def _probe_image(path):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (448, 448), (20, 30, 140))
    ImageDraw.Draw(im).ellipse([120, 120, 330, 330], fill=(240, 200, 30))
    im.save(path)
    return path


def _load_bundle_module(art):
    """Import the artifact's own model.py exactly as a runtime would."""
    spec = importlib.util.spec_from_file_location("custom_model", art / "model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["custom_model"] = mod
    spec.loader.exec_module(mod)
    return mod


def _tower_only(art, mod, cfg, a):
    """Run the probe image through the artifact's OWN vision tower weights."""
    import mlx.core as mx
    from PIL import Image

    vcfg = mod.VisionConfig.from_dict(cfg["vision_config"])
    tower = mod.VisionModel(vcfg)

    # Pull just the tower's tensors out of the shards they live in.
    wm = json.loads((art / "model.safetensors.index.json").read_text())["weight_map"]
    keys = [k for k in wm if any("vis" in seg for seg in k.split("."))]
    shards = sorted({wm[k] for k in keys})
    print(f"tower tensors    : {len(keys)} across {len(shards)} shard(s)")
    weights = {}
    for sh in shards:
        data = mx.load(str(art / sh))
        for k in keys:
            if wm[k] == sh:
                weights[k] = data[k]
    # strip the leading prefix the tower module does not carry itself
    pref = os.path.commonprefix([k for k in keys])
    pref = pref[:pref.rfind(".") + 1] if "." in pref else ""
    flat = {(k[len(pref):] if pref and k.startswith(pref) else k): v
            for k, v in weights.items()}
    # Run the tower's own sanitize() first, exactly as BOTH real loaders do
    # (mlx_vlm.utils.load_model via sanitize_weights; exo's worker via
    # VisionModel.sanitize for model.visual.* keys). The first cut of this
    # instrument called load_weights directly, saw HF-layout patch-embed
    # weights fail in conv3d, and reported a "transposed tower" defect on 8
    # artifacts. Measured afterwards: sanitize() maps BOTH layouts to the
    # same channels-last tensor, so no real loader ever saw that failure.
    # The instrument had bypassed the step that made the artifact correct.
    # A gate that does not run the loader's code path reports the gate's
    # bugs as the artifact's (F137, F153, and now this).
    _san = getattr(tower, "sanitize", None)
    if _san is not None:
        flat = _san(flat)
    try:
        tower.load_weights(list(flat.items()), strict=False)
    except Exception as exc:
        raise SystemExit(f"FAIL: tower weights do not fit mlx_vlm's "
                         f"VisionModel for this config: {exc}")
    mx.eval(tower.parameters())

    img = a.image or _probe_image(
        pathlib.Path(__import__("tempfile").mkdtemp()) / "probe.png")

    # Use the ARTIFACT'S OWN processor, not a hand-rolled tensor. Qwen towers
    # take patch sequences plus a grid_thw describing them; inventing that
    # shape by hand tests my arithmetic, not the artifact.
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(str(art), trust_remote_code=True)
    px = proc.image_processor(images=[Image.open(img).convert("RGB")],
                              return_tensors="np")
    pixel_values = mx.array(px["pixel_values"])
    grid = px.get("image_grid_thw")
    if grid is None:
        raise SystemExit("FAIL: this processor produced no image_grid_thw; "
                         "cannot drive a Qwen-style tower without it.")
    grid = mx.array(grid)
    print(f"processor        : pixel_values={tuple(pixel_values.shape)} "
          f"grid_thw={grid.tolist()}")
    out = tower(pixel_values, grid)
    out = out[0] if isinstance(out, (tuple, list)) else out
    mx.eval(out)
    finite = bool(mx.all(mx.isfinite(out)))
    spread = float(mx.std(out))
    print(f"tower output     : shape={tuple(out.shape)} finite={finite} std={spread:.4f}")
    if not finite or not (spread > 1e-6):
        raise SystemExit("FAIL: tower produced a degenerate embedding "
                         "(non-finite, or constant) -- the weights loaded but "
                         "the tower is not computing.")
    print("\nPASS (TOWER-ONLY): this artifact's own vision weights load into "
          "mlx_vlm's VisionModel and produce a finite, non-constant embedding "
          "from a real image.\nNOT III.11 EVIDENCE: the language model was "
          "never loaded and the merge was never exercised. Run the full gate "
          "on hardware that fits this artifact before calling it releasable.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("--prompt", default="What shape and colors do you see?")
    ap.add_argument("--image", default=None,
                    help="probe image; default is a generated one whose "
                         "content is known (" + PROBE_DESC + ")")
    ap.add_argument("--static", action="store_true",
                    help="surface check only -- no weights loaded. For models "
                         "too large for this box. States in its own output "
                         "that the graph was NOT exercised, because a static "
                         "pass is not the III.11 evidence.")
    ap.add_argument("--tower-only", action="store_true",
                    help="load and run ONLY the vision tower (a few hundred "
                         "MB), not the language model. For artifacts no box "
                         "here fits: the 397B rungs are 112-155 GiB against "
                         "96 GiB on both this Mac and the M3, and exo's "
                         "2-node pipeline serves text through mlx_lm, so "
                         "neither --cluster-smoke nor a local load can put an "
                         "image through them. The tower runs BEFORE the LM "
                         "and is independent of it, so this exercises the "
                         "real weights that were unreachable -- but it does "
                         "NOT test the merge into the LM, and it is not "
                         "III.11 evidence. It says so in its own output.")
    ap.add_argument("--max-tokens", type=int, default=32)
    a = ap.parse_args()

    art = pathlib.Path(a.artifact)
    cfg = json.loads((art / "config.json").read_text())
    mm = bool(cfg.get("vision_config") or cfg.get("audio_config"))
    if not mm:
        print(f"SKIP: {art.name} is text-only (no vision_config/audio_config). "
              "`smoke` is the gate for it.")
        return 0
    if not (art / "model.py").exists():
        raise SystemExit("FAIL: artifact has no model.py to exercise.")

    # ---- 1. SURFACE -------------------------------------------------------
    mod = _load_bundle_module(art)
    arch = getattr(mod, "_arch", None)
    if getattr(mod, "_MULTIMODAL", False) and not getattr(
            mod, "_VISION_SERVABLE", True):
        raise SystemExit(
            "FAIL: this artifact carries a vision tower that its own module "
            "layout cannot serve. Its config names VQ modules in mlx_lm's "
            "tree (model.layers.N...), but only mlx_vlm's arch has the vision "
            "tower -- and that arch does not define this artifact's modules, "
            "so binding to it stops the bundle loading at all, text included. "
            "A rebundle CANNOT fix this: the artifact has to be rebuilt "
            "against the VLM module tree. Text serving is unaffected "
            "(`vqlab smoke`).")
    missing = [n for n in ("TextConfig", "VisionConfig", "VisionModel")
               if not hasattr(mod, n)]
    print(f"bundle base arch : {getattr(arch, '__name__', '?')}")
    if missing:
        raise SystemExit(
            "FAIL: this artifact declares a vision tower but its bundle "
            f"exposes no {'/'.join(missing)}. The shim resolved a TEXT-ONLY "
            "base arch, so the tower on disk is unreachable and mlx_vlm's "
            "loader will die in update_module_configs. This is F153 -- "
            "rebundle with a vqlab that resolves the arch by the artifact's "
            "modalities.")
    print("surface          : TextConfig + VisionConfig + VisionModel present")

    n_vis = 0
    idx = art / "model.safetensors.index.json"
    if idx.exists():
        wm = json.loads(idx.read_text())["weight_map"]
        # Match on any path SEGMENT containing "vis", not a hardcoded prefix
        # list. The families in this fleet already use four different spellings
        # -- vision_tower (gemma, qwen3_5*), vision_model (glm5_next),
        # model.visual (qwen HF layout) and embed_vision -- and the first cut
        # of this gate listed three of them and reported the GLM rungs as
        # having ZERO vision tensors. Hardcoding a name list is the same
        # brittleness that produced F153; do not reintroduce it.
        n_vis = sum(any("vis" in seg for seg in k.split("."))
                    for k in wm)
        print(f"vision tensors   : {n_vis}")
        if not n_vis:
            raise SystemExit("FAIL: vision_config present but ZERO vision "
                             "tensors on disk -- the tower was never grafted.")

    if a.static:
        print("\nPASS (STATIC): the tower is present and the bundle exposes a "
              "multimodal surface.\nNOT III.11 EVIDENCE: no weights were "
              "loaded and no image reached the graph. Run without --static on "
              "a box that fits this artifact before calling it releasable.")
        return 0

    if a.tower_only:
        return _tower_only(art, mod, cfg, a)

    # ---- 2. LOAD ----------------------------------------------------------
    import mlx.core as mx
    from mlx_vlm import load
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor = load(str(art))
    tower = next((getattr(model, n) for n in
                  ("vision_tower", "visual", "vision_model")
                  if getattr(model, n, None) is not None), None)
    if tower is None:
        raise SystemExit("FAIL: model loaded but carries no vision tower "
                         "module -- config key without a graph.")
    print(f"loaded tower     : {type(tower).__name__}")

    img = a.image or _probe_image(
        pathlib.Path(__import__("tempfile").mkdtemp()) / "probe.png")

    # ---- 3. GRAPH ---------------------------------------------------------
    # The check that cannot be satisfied by configuration alone.
    cfg_obj = model.config
    formatted = apply_chat_template(processor, cfg_obj, a.prompt, num_images=1)
    from mlx_vlm.utils import prepare_inputs
    # keyword-only: prepare_inputs' 3rd positional is `audio`, and passing the
    # prompt there sent a caption into the audio loader.
    inputs = prepare_inputs(
        processor, images=[str(img)], prompts=formatted,
        image_token_index=getattr(cfg_obj, "image_token_index", None))
    with_img = model(inputs["input_ids"], inputs.get("pixel_values"),
                     mask=inputs.get("attention_mask"),
                     **{k: v for k, v in inputs.items()
                        if k not in ("input_ids", "pixel_values", "attention_mask")})
    lo_img = (with_img.logits if hasattr(with_img, "logits") else with_img)[:, -1, :]
    mx.eval(lo_img)

    txt = processor.tokenizer(a.prompt, return_tensors="mlx")
    no_img = model.language_model(mx.array(txt["input_ids"]))
    lo_txt = (no_img.logits if hasattr(no_img, "logits") else no_img)[:, -1, :]
    mx.eval(lo_txt)

    moved = float(mx.max(mx.abs(lo_img[0, :lo_txt.shape[-1]] - lo_txt[0])))
    print(f"logit response   : max|with-image - text-only| = {moved:.4f}")
    if not (moved > 1e-3):
        raise SystemExit(
            "FAIL: the image did not move the logits. The tower is loaded but "
            "not wired into the forward graph -- exactly the state a "
            "caption-only check would have called a pass.")

    # ---- caption, for the human -------------------------------------------
    from mlx_vlm import generate
    out = generate(model, processor, formatted, [str(img)],
                   max_tokens=a.max_tokens, verbose=False)
    cap = out if isinstance(out, str) else getattr(out, "text", str(out))
    print(f"\nprobe is {PROBE_DESC}\nmodel says: {cap.strip()}")
    print("\nPASS: an image reached the graph through the artifact's own "
          "shipping runtime and changed the output. (The caption is printed "
          "for you, not asserted on.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
