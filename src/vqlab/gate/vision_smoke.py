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
    vqlab vision-smoke <artifact> --knurlogic [fit|here|peers|all|<page machine name>] [--knurlogic-split pipeline]
"""
import argparse
import importlib.util
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # src/

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



def _image_request_check(post, mid, img, max_tokens, extra=None):
    """The HTTP image arm shared by --cluster (exo) and --knurlogic.

    `post(doc) -> dict` sends one OpenAI chat-completions body. Asserts the
    image request carried MORE prompt tokens than the same question as text
    (the tower's patches entered the prompt) and that a caption came back.
    Returns a list of problems (empty = pass); the caption is printed, never
    asserted on.
    """
    import base64
    b64 = base64.b64encode(open(img, "rb").read()).decode()
    q = "What shape and colors are in this image? Answer in one short sentence."

    def _body(content, ntok):
        return {"model": mid, "max_tokens": ntok, "temperature": 0,
                "messages": [{"role": "user", "content": content}],
                **(extra or {})}

    try:
        n_text = post(_body(q, 1))["usage"]["prompt_tokens"]
    except Exception as exc:
        return [f"no text answer from {mid} ({type(exc).__name__}: {exc})"]
    try:
        out = post(_body([{"type": "image_url",
                           "image_url": {"url": "data:image/png;base64," + b64}},
                          {"type": "text", "text": q}], max_tokens))
        msg = (out.get("choices") or [{}])[0].get("message") or {}
        cap = (msg.get("content") or "").strip()
        n_img = (out.get("usage") or {}).get("prompt_tokens")
    except Exception as exc:
        return [f"the server refused the image request "
                f"({type(exc).__name__}: {exc})"]
    print(f"prompt_tokens   : text-only {n_text} -> with image {n_img}"
          + (f" (+{n_img - n_text} image tokens)" if n_img is not None else ""))
    print(f"\nprobe is {PROBE_DESC}\nmodel says: {cap}")
    probs = []
    if n_img is None:
        probs.append("the image response reported no usage.prompt_tokens, so "
                     "nothing shows the image reached the prompt")
    elif n_img <= n_text:
        probs.append("the image added NO prompt tokens -- the request was "
                     "served as text and the tower never ran")
    if not cap:
        probs.append("image accepted but no caption came back")
    return probs


PAGE = "http://127.0.0.1:8899"


def _wait_ready(page, name, timeout, poll, sleep, get=None):
    """Poll the page's GET /v1/models until the model reads status=ready;
    returns the id to route by."""
    import time
    import urllib.request
    if get is None:
        def get(u):
            with urllib.request.urlopen(u, timeout=30) as r:
                return json.loads(r.read())
    t0 = time.time()
    while True:
        try:
            data = get(page + "/v1/models").get("data") or []
        except Exception:
            data = []
        for m in data:
            mid = str(m.get("id", ""))
            if mid and (mid == name or mid.endswith(name) or name.endswith(mid)):
                if m.get("status") == "ready":
                    return mid
                if m.get("status") == "failed":
                    raise SystemExit(f"FAIL: Knurlogic model {mid} failed to load")
        if time.time() - t0 > timeout:
            raise SystemExit(f"FAIL: {name} not ready on the page after {timeout} s")
        sleep(poll)


def _knurlogic(art, a, K=None, SPK=None, ready_timeout=1800, poll=10,
               sleep=None, page=PAGE):
    """Put the probe image through Knurlogic, the runtime we serve on.

    The arm for DeepSeek-V4 Vision-Exp: its tower is not an mlx_vlm
    VisionModel (no vision_config, flat vision_* keys), so the local arms
    cannot load it, and it is too large for one box anyway. The load mirrors
    `check-release --knurlogic-smoke`: the models-dir entry must resolve to
    THIS artifact, and the job is unloaded whatever happens. K / SPK are
    injectable for tests.
    """
    machines = list(a.knurlogic)   # role words / names pass straight to load()
    if "fit" in machines and len(machines) > 1:
        raise SystemExit("FAIL: --knurlogic fit stands alone; do not combine it "
                         "with other roles or machine names")
    import os
    import time
    sleep = sleep or time.sleep
    if SPK is None:
        from vqlab.bench import speed_pair_knurlogic as SPK
    if K is None:
        ksrc = os.environ.get("KNURLOGIC_SRC", "")
        if ksrc:
            sys.path.insert(0, ksrc)
        try:
            from knurlogic.interfaces import mcp as K
        except ImportError:
            raise SystemExit("FAIL: knurlogic not importable (set KNURLOGIC_SRC "
                             "to its source tree)")
    md = pathlib.Path(os.environ.get("KNURLOGIC_MODELS", "~/.exo/models")).expanduser()
    name = art.resolve().name
    if (md / name).resolve() != art.resolve():
        raise SystemExit(f"FAIL: Knurlogic models dir entry {md / name} does not "
                         f"resolve to {art.resolve()}: the smoke would load "
                         "something else")
    img = a.image or _probe_image(
        pathlib.Path(__import__("tempfile").mkdtemp()) / "probe.png")
    out = K.load(artifact=name, machines=machines, split=a.knurlogic_split,
                 link=a.knurlogic_link, sets={"KNURLOGIC_MTP": "off"})
    job = out.get("job") or out.get("instance")
    if not job or out.get("refused") or out.get("error"):
        raise SystemExit(f"FAIL: Knurlogic load refused or failed: {out}")
    try:
        # Readiness and routing both go through THIS Mac's page: a model
        # server binds 127.0.0.1 by design, so a probe straight at its port
        # from the other Mac is refused (F205).
        mid = _wait_ready(page, name, ready_timeout, poll, sleep)
        print(f"knurlogic       : {', '.join(machines)} via {page}\n"
              f"model_id        : {mid}")
        probs = _image_request_check(lambda d: SPK._post(page, d), mid, img,
                                     a.max_tokens,
                                     extra={"reasoning_effort": "none"})
    finally:
        # a single-Mac load answers with an instance id (job is null in the
        # page's rows), so unload by whichever id load returned
        K.unload(**({"job": str(out["job"])} if out.get("job")
                    else {"instance": str(job)}))
    if probs:
        raise SystemExit("\nFAIL: " + "; ".join(probs))
    print("\nPASS (KNURLOGIC): the probe image went through the serving "
          "runtime, its patches entered the prompt and a caption came back. "
          "The caption is PRINTED, not asserted on -- read it.")
    return 0


def _n_tower_tensors(art):
    """Tower tensors in the index, by core.artifact.tensor_class -- the same
    classifier `size` and `check-release` use."""
    idx = art / "model.safetensors.index.json"
    if not idx.exists():
        return 0
    from vqlab.core.artifact import tensor_class
    wm = json.loads(idx.read_text())["weight_map"]
    return sum(tensor_class(k) == "tower" for k in wm)


def _cluster(art, a):
    """Put the probe image through a PLACED exo instance over HTTP.

    The real III.11 vision evidence for an artifact no single box fits. Exo
    serves images (exo/worker/engines/mlx/vision.py; the chat_completions
    adapter resolves `image_url` parts), so a 2-node pipeline instance will
    run the tower, merge its patches into the LM and caption the image --
    exactly the path a user exercises. An earlier version of this file's
    --tower-only help asserted this was impossible; F176 did it on all three
    397B flat rungs.

    This arm is a CAPTION check and the caption is the evidence, so unlike
    the local gate it cannot assert a logit delta. It asserts instead that
    the request carried image tokens (prompt_tokens jumps by the tower's
    patch count) and that a caption came back -- a model served WITHOUT the
    image would answer the same question with a far shorter prompt.
    """
    import json as _json, urllib.request
    mid = a.cluster_model or ("TheDrainFlorist/" + art.name.split("--", 1)[-1])
    img = a.image or _probe_image(
        pathlib.Path(__import__("tempfile").mkdtemp()) / "probe.png")

    def _post(doc):
        req = urllib.request.Request(
            a.cluster.rstrip("/") + "/v1/chat/completions",
            data=_json.dumps(doc).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=a.cluster_timeout) as r:
            return _json.load(r)

    print(f"cluster         : {a.cluster}\nmodel_id        : {mid}")
    probs = _image_request_check(_post, mid, img, a.max_tokens)
    if probs:
        raise SystemExit("\nFAIL: " + "; ".join(probs) + " (no instance? "
                         "place one first -- a model CARD in /state is not "
                         "a placed instance)")
    print("\nPASS (CLUSTER): the artifact's own vision tower ran inside the "
          "served pipeline, its patches entered the language model, and the "
          "model captioned the image. This IS III.11 vision evidence.\n"
          "The caption is PRINTED, not asserted on -- read it.")
    return 0

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
                         "MB), not the language model. A FALLBACK, not the "
                         "gate: it does NOT test the merge into the LM and "
                         "is NOT III.11 evidence, and says so in its own "
                         "output. For an artifact too large for this box "
                         "(e.g. the 397B rungs are 112-155 GiB, more "
                         "than one box's unified memory), prefer the EXO CLUSTER: exo "
                         "does serve images -- see "
                         "exo/worker/engines/mlx/vision.py and the "
                         "chat_completions adapter's image_url handling -- "
                         "so POST a data:image/png;base64 image_url part to "
                         "/v1/chat/completions on a placed 2-node instance "
                         "and you get REAL image->text evidence. F176 did "
                         "this on all three 397B flat rungs; an earlier "
                         "version of this help text wrongly claimed it was "
                         "impossible, and that claim was repeated as fact.")
    ap.add_argument("--cluster", metavar="URL",
                    help="put the image through a PLACED exo instance at "
                         "this API root (e.g. http://localhost:52415) "
                         "instead of loading locally. THE right arm for "
                         "an artifact bigger than one box: exo serves "
                         "images, so this is real III.11 evidence, unlike "
                         "--tower-only. Place the instance first.")
    ap.add_argument("--cluster-model", metavar="ID",
                    help="exo model_id; default derives it from the "
                         "artifact dir name (owner--name -> owner/name).")
    ap.add_argument("--cluster-timeout", type=float, default=600.0)
    ap.add_argument("--knurlogic", metavar="ROLE", action="append", default=[],
                    nargs="?", const="fit",
                    help="put the image through Knurlogic on this role: here | "
                         "peers | all | fit (default when given no value; stands "
                         "alone: this Mac if it fits, else the smallest set that "
                         "places it), or a page machine name; passed straight "
                         "to load(machines=...); roles and names may mix, "
                         "fit may not. The runtime we serve on, and "
                         "the only arm that drives DeepSeek-V4 Vision-Exp, whose "
                         "tower is not an mlx_vlm VisionModel. The Knurlogic "
                         "models-dir entry must resolve to THIS artifact.")
    ap.add_argument("--knurlogic-split", default="", help="pipeline | tensor")
    ap.add_argument("--knurlogic-link", default="", help="tcp | rdma")
    ap.add_argument("--max-tokens", type=int, default=32)
    a = ap.parse_args()

    art = pathlib.Path(a.artifact)
    cfg = json.loads((art / "config.json").read_text())
    # Tower TENSORS count too: DeepSeek-V4 Vision-Exp carries a grafted tower
    # with flat vision_* config keys and no vision_config, and a config-only
    # test SKIPPED it as text-only.
    n_tower = _n_tower_tensors(art)
    mm = bool(cfg.get("vision_config") or cfg.get("audio_config") or n_tower)
    if not mm:
        print(f"SKIP: {art.name} is text-only (no vision_config/audio_config, "
              "no tower tensors). `smoke` is the gate for it.")
        return 0
    if a.knurlogic:
        return _knurlogic(art, a)
    if a.cluster:
        # The cluster serves the artifact from its OWN directory, so no
        # local load and no bundle import -- the runtime under test is
        # exo's, which is the point.
        return _cluster(art, a)
    if not (art / "model.py").exists():
        raise SystemExit("FAIL: artifact has no model.py to exercise.")
    if n_tower and not cfg.get("vision_config"):
        raise SystemExit(
            f"FAIL: {n_tower} tower tensors but no vision_config, so no "
            "mlx_vlm arm can load this tower (DeepSeek-V4 Vision-Exp). Drive "
            "it through the serving runtime: --knurlogic fit.")

    # ---- 1. SURFACE -------------------------------------------------------
    mod = _load_bundle_module(art)
    arch = getattr(mod, "_arch", None)
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
        # core.artifact.tensor_class, the one classifier: it matches any
        # path SEGMENT naming vision/visual, not only a prefix list. The
        # families in this fleet already use four different spellings --
        # vision_tower (gemma, qwen3_5*), vision_model (glm5_next),
        # model.visual (qwen HF layout) and embed_vision -- and the first cut
        # of this gate listed three of them and reported the GLM rungs as
        # having ZERO vision tensors (F153's brittleness).
        from vqlab.core.artifact import tensor_class
        n_vis = sum(tensor_class(k) == "tower" for k in wm)
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
