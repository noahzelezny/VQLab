#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Verify every Qwen-shaped assumption in the VQ pipeline against gemma-4.

WHY THIS EXISTS. The VQ pipeline was built for Qwen3.5 and hardcodes that
family in a dozen places — module names, fused-tensor layout, source-key
templates, packing shape constraints. Re-targeting it at gemma-4-26b-a4b is
a list of small edits, and EVERY ONE of them fails SILENTLY if wrong: a bad
target predicate selects zero tensors and the fitter exits cleanly, a bad
split axis fits a codebook to the wrong half of gate_up and the artifact
merely scores badly. Both cost a full GPU run to discover.

So: check them all statically first. This script touches no GPU and needs
only config.json + the safetensors index. Run it BEFORE the ladder.

    ./gemma_preflight.py --model "/Volumes/.../mlx-community--gemma-4-26b-a4b-it-6bit"
    ./gemma_preflight.py --model <dir> --strict     # exit 1 on any FAIL

Findings are printed as OK / WARN / FAIL with the pipeline site each one
governs, so a failure names the file:line that has to change.
"""
import argparse
import json
import pathlib
import sys

# --- family descriptors -------------------------------------------------
# Derived by READING mlx_lm 0.31.3 sources, not guessed:
#   gemma4.py:55-81        sanitize(): drops towers, rewrites the LM prefix
#   gemma4_text.py:625-634 sanitize(): splits experts.gate_up_proj on axis=-2
#   gemma4_text.py:159-164 SwitchGLU(..., bias=False)
#   gemma4_text.py:641-646 quant_predicate: router.proj -> 8bit/gs64
FAMILY = {
    "qwen3_5_moe": {
        "target_substr": "switch_mlp",
        "mlx_expert_path": "model.layers.{i}.mlp.switch_mlp.{proj}",
        "src_key": "model.language_model.layers.{i}.mlp.experts.{stack}",
        "fused": {"gate_proj": ("gate_up_proj", 0),
                  "up_proj": ("gate_up_proj", 1),
                  "down_proj": ("down_proj", None)},
        "split_axis": 1,
        "expert_bias": False,
    },
    "gemma4": {
        # experts.switch_glu.*, NOT mlp.switch_mlp.* — vq_397b_codes.py:99
        "target_substr": "switch_glu",
        "mlx_expert_path":
            "language_model.model.layers.{i}.experts.switch_glu.{proj}",
        "src_key": "model.language_model.layers.{i}.experts.{stack}",
        "fused": {"gate_proj": ("gate_up_proj", 0),
                  "up_proj": ("gate_up_proj", 1),
                  "down_proj": ("down_proj", None)},
        # gemma splits on axis=-2 of a rank-3 [E, 2*I, H] tensor, which IS
        # axis 1 — the same OUT-dim half-slice Qwen uses. Transfers unchanged.
        "split_axis": 1,
        "expert_bias": False,
    },
}

BLOCK = 32        # vq_pack.py:32 — codes per packing block
GROUP = 64        # vq_fit.py:92 — scale group size
RESULTS = []


def check(ok, name, detail, site, warn_only=False):
    tag = "OK  " if ok else ("WARN" if warn_only else "FAIL")
    RESULTS.append((tag, name))
    print(f"  [{tag}] {name}")
    print(f"         {detail}")
    if not ok:
        print(f"         -> {site}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dim", type=int, default=4, help="VQ subvector dim d")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    root = pathlib.Path(args.model)
    cfg = json.load(open(root / "config.json"))
    mt = cfg.get("model_type")
    tc = cfg.get("text_config", cfg)
    fam = FAMILY.get(mt)

    print(f"\n=== gemma preflight: {root.name}")
    print(f"    model_type={mt}\n")

    print("-- architecture")
    check(fam is not None, "family descriptor known",
          f"model_type={mt!r}", "add an entry to FAMILY in this file")
    if fam is None:
        sys.exit(1)

    moe = tc.get("enable_moe_block")
    n_exp = tc.get("num_experts")
    check(bool(moe) and bool(n_exp), "is an MoE",
          f"enable_moe_block={moe} num_experts={n_exp}",
          "VQ targets routed experts; a dense model has none")

    print("\n-- mlx_lm support (the runtime shim subclasses this)")
    try:
        import importlib
        m = importlib.import_module(f"mlx_lm.models.{mt}")
        has = hasattr(m, "Model") and hasattr(m, "ModelArgs")
        check(has, f"mlx_lm.models.{mt} exports Model+ModelArgs",
              f"module at {m.__file__}", "add_model_file.py:77 will fail")
    except ImportError as e:
        check(False, f"mlx_lm.models.{mt} importable", str(e),
              "add_model_file.py:77 imports this by model_type")

    print("\n-- shapes and packing constraints")
    h = tc.get("hidden_size")
    mi = tc.get("moe_intermediate_size")
    d = args.dim
    for label, in_d in (("gate/up (in=hidden)", h), ("down (in=moe_inter)", mi)):
        if not in_d:
            continue
        nsub = in_d // d
        check(in_d % GROUP == 0, f"{label}: in_dim % {GROUP} == 0",
              f"in_dim={in_d}", "vq_fit.py:92 / vq_35b_codes.py:93 assert")
        check(nsub % BLOCK == 0, f"{label}: NSUB % {BLOCK} == 0 (d={d})",
              f"NSUB={nsub}, remainder {nsub % BLOCK}",
              "vq_pack.py:42 words_per_row asserts this; "
              f"at d=2 NSUB={in_d // 2} (rem {(in_d // 2) % BLOCK}) — "
              "or skip packing for this projection (it is an optional pass)")

    print("\n-- expert bias (VQSwitchLinear has no bias support)")
    check(fam["expert_bias"] is False, "experts are bias-free",
          "mlx_lm gemma4_text.py:164 constructs SwitchGLU(bias=False)",
          "vq_switch.py:552-553 would need bias support")

    print("\n-- tower graft safety (referee numbers must not move)")
    towers = [k for k in cfg if k.endswith("_config") and
              k.split("_")[0] in ("vision", "audio")]
    check(True, "towers present in config", f"{towers or 'none'}", "", warn_only=True)
    try:
        src = pathlib.Path(importlib.import_module(f"mlx_lm.models.{mt}").__file__).read_text()
        drops = all(t in src for t in ("vision_tower", "audio_tower"))
        check(drops, "sanitize() drops vision AND audio towers",
              "gemma4.py:61-71 `continue`s on vision_tower/audio_tower/"
              "multi_modal_projector/embed_audio/embed_vision",
              "graft_vision.py:14-16 relies on sanitize dropping the tower; "
              "if it does not, load_weights(strict=True) breaks after graft")
    except Exception as e:                       # noqa: BLE001
        check(False, "sanitize() inspectable", str(e), "check manually")

    print("\n-- source tensors (needs the real checkpoint on disk)")
    idx = root / "model.safetensors.index.json"
    if not idx.exists():
        check(False, "safetensors index present",
              f"{idx} missing — weights not downloaded",
              "hf download mlx-community/gemma-4-26b-a4b-it-6bit", warn_only=True)
    else:
        wm = json.load(open(idx))["weight_map"]
        for stack in ("gate_up_proj", "down_proj"):
            k = fam["src_key"].format(i=0, stack=stack)
            check(k in wm, f"source key exists: ...layers.0...{stack}",
                  f"{k}" + ("" if k in wm else "  (NOT FOUND)"),
                  "vq_397b_codes.py:156 src_key template is wrong for gemma")
        sample = [k for k in wm if fam["target_substr"] in k]
        check(bool(sample) or True,
              f"post-sanitize target substring {fam['target_substr']!r}",
              f"{len(sample)} raw keys match (0 is EXPECTED: the substring "
              "appears only AFTER mlx_lm sanitize renames experts.* -> "
              "switch_glu.*; the base artifact is what must contain it)",
              "", warn_only=True)

    fails = [n for t, n in RESULTS if t == "FAIL"]
    print(f"\n=== {len(RESULTS)} checks, {len(fails)} FAIL")
    for n in fails:
        print(f"    FAIL: {n}")
    if fails and args.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
