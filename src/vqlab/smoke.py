#!/usr/bin/env python3
"""vqlab smoke — generate one token through the runtime the artifact SHIPS.

This is the gate the rest of the pipeline cannot replace. Every byte-level
check reads bytes; none runs the model. An artifact in this project passed the
outlier gate, the release checks, the vision check AND the scoring referee,
then raised on its first real forward pass — because the referee scores
through the reference decode path while serving uses the fused kernel. A rung
can score normally and be unable to serve.

It also NAMES THE RUNTIME THAT ACTUALLY RESOLVED. Which copy executes — the
artifact's bundled model.py, site-packages, or a repo checkout — depends on
the loader and the environment, and assuming it has produced wrong
conclusions in both directions. So this prints `__file__` for the modules that
really loaded rather than asserting which one should have.

Generation needs the WHOLE model resident: this refuses to run an artifact
larger than the box unless you override, because a smoke that thrashes swap
produces no verdict.

    vqlab smoke <artifact> [--prompt TEXT] [--max-tokens N] [--headroom F]
                           [--skip-preflight]
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vqlab smoke",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("artifact")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-tokens", type=int, default=4,
                    help="tokens to generate. One is enough to satisfy the "
                         "rule; the default of 4 also catches a model that "
                         "produces exactly one token and then dies.")
    ap.add_argument("--headroom", type=float, default=0.90)
    ap.add_argument("--skip-preflight", action="store_true",
                    help="run even if the artifact may not fit in RAM. You "
                         "will get swap thrash and no verdict; this exists "
                         "for boxes whose memory the preflight misreads.")
    a = ap.parse_args(argv)
    art = pathlib.Path(a.artifact)
    if not (art / "config.json").exists():
        print(f"FAIL: {art} has no config.json")
        return 1

    if not a.skip_preflight:
        p = subprocess.run([sys.executable, str(HERE / "preflight_ram.py"),
                            str(art), "--headroom", str(a.headroom)])
        if p.returncode != 0:
            print("FAIL: artifact does not fit in RAM — a generation smoke "
                  "CANNOT produce a verdict here. Use a bigger box, or "
                  "--skip-preflight if you know better than the check.")
            return 1

    # Runtime dispatch (Noah's ruling 08-29): the artifact's model_type
    # resolves to a family, the family names its runtime. mlx_lm families
    # take exactly the path this tool always took. Both runtimes honour the
    # in-checkpoint model.py bundle, so "generate through the runtime the
    # artifact SHIPS" holds under either.
    import json as _json
    sys.path.insert(0, str(HERE))
    import runtime_load
    _cfg = _json.load(open(art / "config.json"))
    _mt = _cfg.get("model_type") or \
        _cfg.get("text_config", {}).get("model_type")
    _fam = runtime_load.family_for_model_type(_mt)
    _rt = runtime_load.runtime_for(_fam) if _fam else "mlx_lm"

    print(f"loading {art} ... (runtime {_rt})", flush=True)
    if _rt == "mlx_vlm":
        # PROVISIONAL: written from a source read of mlx-vlm main, never
        # executed here. Text-only smoke of a VLM artifact; the vision path
        # gets its own gate when GLM vision is actually exercised.
        from mlx_vlm import generate as vlm_generate
        from mlx_vlm.utils import load as vlm_load
        model, processor = vlm_load(str(art))
        tokenizer = getattr(processor, "tokenizer", processor)
    else:
        from mlx_lm.utils import load
        from mlx_lm import generate
        # VQ artifacts ship their runtime in-checkpoint; loading it is the point.
        model, tokenizer = load(str(art), trust_remote_code=True)
    print(runtime_load.resolved_runtime_note(model))

    # III.13: instrument the import, never assume which copy runs.
    print("\nRESOLVED RUNTIME (the copy that actually loaded):")
    seen, vq_modules = set(), 0
    for name, mod in model.named_modules() if hasattr(model, "named_modules") \
            else []:
        cls = type(mod)
        if cls.__name__.startswith("VQ"):
            vq_modules += 1
            origin = sys.modules.get(cls.__module__)
            path = getattr(origin, "__file__", None)
            if path is None:
                import inspect
                try:
                    path = inspect.getfile(cls)
                except TypeError:
                    path = f"<module {cls.__module__!r} — exec'd, no file>"
            key = (cls.__name__, path)
            if key not in seen:
                seen.add(key)
                print(f"  {cls.__name__:18s} <- {path}")
    if not seen:
        print("  (no VQ modules found — is this a VQ artifact?)")
    print(f"  {vq_modules} VQ module(s) instantiated")

    # Use the model's chat template when it has one. An instruct model fed
    # a raw prompt can emit a degenerate repetition loop that LOOKS like
    # quantization damage — measured: a pure-affine control produced the
    # identical loop on the same raw prompt, and the VQ artifact answered
    # correctly once the template was applied. A smoke that alarms on the
    # instrument rather than the artifact wastes exactly the trust it exists
    # to build.
    prompt = a.prompt
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": a.prompt}],
            tokenize=False, add_generation_prompt=True)
        print("(chat template applied)")
    print(f"\ngenerating {a.max_tokens} token(s) through the fused path ...",
          flush=True)
    try:
        if _rt == "mlx_vlm":
            # PROVISIONAL text-only generate; mlx_vlm's generate wants the
            # processor and returns a result object or str per version.
            text = vlm_generate(model, processor, prompt=prompt,
                                max_tokens=a.max_tokens, verbose=False)
            text = getattr(text, "text", text)
        else:
            text = generate(model, tokenizer, prompt=prompt,
                            max_tokens=a.max_tokens, verbose=False)
    except Exception as e:
        print(f"\nFAIL: the artifact could not generate: "
              f"{type(e).__name__}: {e}\n\n"
              "This is the failure every byte-level gate is blind to. Do NOT "
              "release this artifact; a score from it describes a model that "
              "cannot serve.")
        return 1
    if not text or not text.strip():
        print("\nFAIL: generation returned empty text.")
        return 1
    print(f"\n  {a.prompt!r} -> {text!r}")
    print("\nPASS: artifact generated through its shipping runtime. "
          "Record the resolved paths above in any runtime-dependent claim.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
