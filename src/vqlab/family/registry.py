"""Shared model-family registry — the single source of truth.

MoE/verify families (FAMILY): tensor-name templates and projection maps for
the MoE fitter and the outlier gate. Dense fitter families (DENSE_FAMILIES):
key template + default layer range for fit_dense_vq. Historically this dict
lived inline in fit_moe.py, was text-scraped-and-exec'd by
verify_artifact.py, and was duplicated (with a diverged name) in
fit_dense_vq.py; one table, imported everywhere, ends that.
"""
import re

from vqlab.core.source.teacher_profiles import profiles_in, read_profile

FAMILY = {
    # "recipe": the Recipe page's build chain for the family, "moe" or "dense"
    # (plan/recipe_queue.py); absent = the family builds through the CLI only.
    # What the chain does differently per family, as the published builds
    # were made:
    #   "ple": {"key_regex", "group"}  n-gram tables VQ-fitted on their own
    #          (fit-ple -> splice-ple -> pack-ple; Flash-Next)
    #   "tower": "base"  the vision tower rides the struct base bf16, so it
    #          is never grafted (GLM)
    #   "embed": {"key", "group"}  the VQ target is this one embedding table,
    #          nothing else (dense: fit-ple -> build-dense --embed; e4b). Not
    #          "ple": those tables ride an MoE build at their own geometry,
    #          this one IS the build's geometry, spliced as vq_embed
    #   "base_bits": the base width a recipe defaults to, the shipped build's
    #          (base_bits(); 4 when absent)
    #   "incumbent": "org/name"  the quantized Hub build the dense chain
    #          splices into instead of converting its own base (e4b: the
    #          converter's runtime drops the audio tower the shipped build
    #          carries); with "tower": "base" its towers ride it
    "qwen3_5": {
        "recipe": "moe",
        "target_substr": "switch_mlp",
        # HF-layout source: gate and up live FUSED in one [E, 2I, H] stack,
        # taken as halves along the OUT axis.
        "src_key": "model.language_model.layers.{li}.mlp.experts.{key}",
        "proj": {"gate_proj": ("gate_up_proj", 0),
                 "up_proj": ("gate_up_proj", 1),
                 "down_proj": ("down_proj", None)},
    },
    "qwen4_exp": {
        "recipe": "moe",
        "ple": {"key_regex": "ngram_embedding", "group": 32},
        # Qwen3.8-Flash-Next (180B, 10-of-512). VERIFIED IDENTICAL to
        # qwen3_5 on every axis this table encodes (2026-08-28, from the
        # checkpoint index + the mlx-lm PR #1788 module tree): same fused
        # [E, 2I, H] gate_up stack halved along OUT, same HF source key
        # template, same runtime member name (SparseMoeBlock.switch_mlp).
        # All 48 layers are MoE — no dense head despite the Qwen4 lineage.
        # NOT covered here: the per-layer ngram PLE tables (28.4% of
        # params, [*, 160] rows) — those take the PLE fitter, not this
        # one — and the per-layer shared_expert MLP, which stays in the
        # protected budget with attention and the router.
        "target_substr": "switch_mlp",
        "src_key": "model.language_model.layers.{li}.mlp.experts.{key}",
        "proj": {"gate_proj": ("gate_up_proj", 0),
                 "up_proj": ("gate_up_proj", 1),
                 "down_proj": ("down_proj", None)},
    },
    "glm5_next": {
        # GLM-5.3-Flash (320B, 8-of-288 routed + 1 shared, 43 MoE layers on
        # indices 3-45 — 42 main + the MTP layer 45, which carries its own
        # full expert stack; layers 0-2 are dense). MEASURED from the BF16
        # checkpoint's safetensors headers 2026-08-28: experts are UNFUSED per-expert 2D tensors
        # — gate/up [2048, 4096], down [4096, 2048], no gate_up stack, no
        # [E, out, in] stack anywhere. The "{e}" in src_key marks that:
        # expert_src.load_expert_stack gathers experts.{0..E-1} into a
        # stack, discovering E from the index (288 measured; never trust
        # the config's count over the index).
        # target_substr is PROVISIONAL: it matches mlx-vlm's shipped
        # glm5_next (DeepseekV32MoE.switch_mlp = SwitchGLU, sanitize stacks
        # to mlp.switch_mlp.{key}.weight) and mlx-lm's own deepseek_v32
        # convention, but mlx-lm has no glm5_next class yet (checked
        # 2026-08-28) — re-verify the module name against the real class
        # before the first struct base is built.
        # runtime: mlx-lm has no glm5_next class (2026-08-29); mlx-vlm ships
        # one (PR #2030). Flip this ONE field to "mlx_lm" when it lands
        # there — every load site routes through runtime_load.load_for_family.
        # model_type: lets artifact-only tools (smoke, stream_score) resolve
        # this family from a config.json.
        "runtime": "mlx_vlm",
        "model_type": "glm5_next",
        "recipe": "moe", "tower": "base",
        "target_substr": "switch_mlp",
        "src_key": "model.language_model.layers.{li}.mlp.experts.{e}.{key}.weight",
        "proj": {"gate_proj": ("gate_proj", None),
                 "up_proj": ("up_proj", None),
                 "down_proj": ("down_proj", None)},
    },
    "gemma4": {
        "recipe": "moe",
        "target_substr": "switch_glu",
        # mlx-community's gemma bf16 is an MLX-FORMAT conversion, so
        # mlx_lm's sanitize has ALREADY split experts.gate_up_proj into
        # switch_glu.{gate,up}_proj (gemma4_text.py:625-634). There is no
        # fused stack in the checkpoint — verified: 'gate_up_proj' appears
        # in zero keys. So every projection is direct, no half-slicing, and
        # the prefix is language_model.model.* not model.language_model.*.
        "src_key": "language_model.model.layers.{li}.experts.switch_glu.{key}.weight",
        "proj": {"gate_proj": ("gate_proj", None),
                 "up_proj": ("up_proj", None),
                 "down_proj": ("down_proj", None)},
    },
    "gemma4_e4b": {
        # The shipped VQ-PLE build: mlx-community's 8-bit with only the
        # per-layer embedding table VQ'd (d2/K2048, group 64).
        "recipe": "dense", "base_bits": 8, "tower": "base",
        "incumbent": "mlx-community/gemma-4-e4b-it-8bit",
        "embed": {"key": "language_model.model.embed_tokens_per_layer", "group": 64},
        # gemma-4-e4b-it: DENSE mlp (no experts), weights are 2D [OUT, IN].
        # The MLP fields below describe that trio for the tools that read a
        # dense family's MLPs (stream-convert's target_substr, the outlier
        # gate, verify_artifact); the Recipe chain VQs the table only.
        # Consumers must treat a 2D source tensor as [1, OUT, IN] (E=1).
        "target_substr": ".mlp.",
        "src_key": "language_model.model.layers.{li}.mlp.{key}.weight",
        "proj": {"gate_proj": ("gate_proj", None),
                 "up_proj": ("up_proj", None),
                 "down_proj": ("down_proj", None)},
    },
    "qwen3_8_dense": {
        "recipe": "dense",
        # DENSE Qwen3.8-27B. Not an MoE: the mlp trio lives directly on the
        # layer with no .experts./.switch_mlp. segment, and each tensor is 2D
        # ([OUT, IN]) rather than [E, OUT, IN] — verify_artifact adds the E=1
        # axis for dense families. Source is the HF-format bf16 checkpoint,
        # whose prefix is model.language_model.* (the ARTIFACT uses
        # language_model.model.*; build_dense.py owns that remap).
        "target_substr": "mlp",
        "src_key": "model.language_model.layers.{li}.mlp.{key}.weight",
        "proj": {"gate_proj": ("gate_proj", None),
                 "up_proj": ("up_proj", None),
                 "down_proj": ("down_proj", None)},
    },
    "qwen3_5_mlx": {
        # SAME architecture as qwen3_5 (qwen3_5_moe, switch_mlp, shared
        # expert) but sourced from an mlx-community MLX-FORMAT bf16
        # conversion rather than the original HF-format one. Verified on
        # mlx-community/Qwen3.6-35B-A3B-bf16: sanitize already split
        # gate_up_proj (zero fused keys), prefix is language_model.model.*
        # (no leading "model."), and there is no .experts. path segment —
        # the module IS switch_mlp.{gate,up,down}_proj directly. Do not
        # point this at an HF-format source (use "qwen3_5" for that).
        "target_substr": "switch_mlp",
        "src_key": "language_model.model.layers.{li}.mlp.switch_mlp.{key}.weight",
        "proj": {"gate_proj": ("gate_proj", None),
                 "up_proj": ("up_proj", None),
                 "down_proj": ("down_proj", None)},
    },
}

def base_bits(family: str) -> int:
    """The affine base width a recipe for `family` starts at (FAMILY
    "base_bits": the shipped build's), 4 when the family names none."""
    return FAMILY.get(family, {}).get("base_bits", 4)


def embed_regex(family: str) -> str | None:
    """The source-key regex of the family's one VQ'd embedding table (FAMILY
    "embed"), with or without its .weight suffix; None for other families."""
    emb = FAMILY.get(family, {}).get("embed")
    return "^" + re.escape(emb["key"]) + r"(\.weight)?$" if emb else None


def recipe_families(kind: str) -> frozenset:
    """The families the Recipe page builds with the `kind` chain ("moe" or
    "dense"), as each FAMILY entry declares."""
    return frozenset(f for f, e in FAMILY.items() if e.get("recipe") == kind)


# Dense fitter families: (src key template, default layer range "LO-HI").
DENSE_FAMILIES = {
    "gemma4_e4b": ("language_model.model.layers.{li}.mlp.{key}.weight", "0-41"),
    # named to match FAMILY's verify entry; the pre-release name was "qwen3_8"
    "qwen3_8_dense": ("model.language_model.layers.{li}.mlp.{key}.weight", "0-63"),
}

# Dense assembly remap: family -> (fitter module template, base module
# template). The fitter names modules with the SOURCE checkpoint's
# convention because that is what it read; the mlx base uses another. These
# are different strings and nothing downstream catches a mismatch except a
# strict load failure at score time, so build_dense --dry-run asserts
# every fitted module lands on a real base module before bytes are written.
DENSE_REMAP = {
    "qwen3_8_dense": ("model.language_model.layers.{li}.mlp.{key}",
                      "language_model.model.layers.{li}.mlp.{key}"),
    "gemma4_e4b": ("language_model.model.layers.{li}.mlp.{key}",
                   "language_model.model.layers.{li}.mlp.{key}"),
}


# ---------------------------------------------------------------- data entries
# A new family is DATA, not a code change: drop an entry.json into a
# families folder and every tool that reads this registry sees it. That is
# what lets a family be onboarded through `vqlab family-profile
# --write-entry` (or MCP `run`) without editing VQLab or waiting on a PR.
#
#   <repo>/families/<name>/entry.json      checked-in families (read-only,
#                                          present only in a checkout)
#   <families>/<name>/entry.json           a user's own: vqlab.config
#                                          families_dir() ($VQLAB_FAMILIES_DIR,
#                                          else <records>/families)
#
# entry.json: {"kind": "moe", "target_substr", "src_key", "proj":
# {proj: [src, half|null]}, "recipe": "moe" (optional), ...} or
# {"kind": "dense", "src_key", "layers": "LO-HI", "remap": [fit_tmpl,
# base_tmpl]}. A code entry above always wins over a data entry of the same
# name, so a data file cannot silently
# redefine a published family.
def family_dirs():
    """The family folders, checkout first, the user's overriding."""
    from vqlab import config
    return [config.checkout_families_dir(), config.families_dir()]


def iter_profiles():
    """{(family, teacher): profile} for every readable teacher profile: the
    checkout's first, the user's overriding (core/source/teacher_profiles)."""
    return profiles_in(family_dirs())


def teacher_profile(family, teacher):
    """One teacher's profile, by the same precedence as iter_profiles (the
    user's over the checkout's), or None when no readable one exists. A
    name is a folder name, never a path."""
    if any(not n or "/" in n or "\\" in n or n.startswith(".") for n in (family, teacher)):
        return None
    for d in reversed(family_dirs()):
        prof = read_profile(d / family / "teachers" / teacher / "profile.json")
        if prof is not None:
            return prof
    return None


def teacher_record(family: str, teacher: str, name: str) -> dict | None:
    """A teacher's measured record families/<f>/teachers/<t>/<name>, by the
    profile's precedence, or None when none was written (or it is
    unreadable). The records: skipzero.json, its dead rows as sz-pack
    recorded them in a shipped build's config (docs/SKIPZERO.md); mtp.json,
    the MTP head wiring a shipped sidecar was packed with."""
    import json
    if any(not n or "/" in n or "\\" in n or n.startswith(".") for n in (family, teacher)):
        return None
    for d in reversed(family_dirs()):
        f = d / family / "teachers" / teacher / name
        try:
            return json.loads(f.read_text())
        except FileNotFoundError:
            continue
        except (OSError, ValueError):
            return None
    return None


def _load_data_entries():
    import json
    loaded = {}
    for d in family_dirs():
        for f in sorted(d.glob("*/entry.json")) if d.is_dir() else ():
            name = f.parent.name
            if name in FAMILY or name in DENSE_FAMILIES or name in loaded:
                continue
            e = json.loads(f.read_text())
            if e.get("kind", "moe") == "dense":
                DENSE_FAMILIES[name] = (e["src_key"], e.get("layers", "0-0"))
                if e.get("remap"):
                    DENSE_REMAP[name] = tuple(e["remap"])
            else:
                FAMILY[name] = {k: v for k, v in e.items()
                                if k not in ("kind", "_note", "_source")}
                FAMILY[name]["proj"] = {p: tuple(v) for p, v in e["proj"].items()}
            loaded[name] = str(f)
    return loaded


DATA_ENTRIES = _load_data_entries()


def teacher_caches(teacher=None):
    """Registered KL teacher caches: families/<f>/teachers/<t>/teacher_caches.json.

    [{family, teacher, corpus, path, tokens, top_k, full_vocab, ...}], for
    one teacher (its profile name, or a substring of it) or all. A cache
    built outside the onboard/ layout is findable only if it is registered
    here -- the 397B's three full-vocab gate caches were reported missing
    on 2026-09-26 for exactly that reason."""
    import json
    out = []
    for d in family_dirs():
        for f in sorted(d.glob("*/teachers/*/teacher_caches.json")) if d.is_dir() else ():
            t = f.parent.name
            if teacher and teacher.lower() not in t.lower():
                continue
            for c in json.loads(f.read_text()).get("caches", []):
                out.append({"family": f.parent.parent.parent.name, "teacher": t, **c})
    return out


# Families that live as plugins (vqlab/family/) add their fit layouts here.
from vqlab.family import fit_families as _plugin_fit
FAMILY.update(_plugin_fit())


def scorer_variant(model_type: str | None):
    """The family plugin's current scorer variant for model_type (None
    without one, or when the plugin cannot say). Stamped into every cache
    and result by family.version_stamp.build."""
    if not model_type:
        return None
    try:
        from vqlab.family import variant_for
        return variant_for(model_type)
    except Exception:
        return None
