#!/usr/bin/env python3
"""vqlab smoke — generate one token through the runtime the artifact SHIPS.

This is the gate the rest of the pipeline cannot replace. Every byte-level
check reads bytes; none runs the model. An artifact in this project passed the
outlier gate, the release checks, the vision check AND the scoring referee,
then raised on its first real forward pass — because the referee scores
through the reference decode path while serving uses the fused kernel. A rung
can score normally and be unable to serve.

It also NAMES THE RUNTIME THAT ACTUALLY RESOLVED, and since 2026-09-01 it
ASSERTS on it. Which copy executes — the artifact's bundled model.py,
site-packages, or a repo checkout — depends on the loader and the
environment, and assuming it has produced wrong conclusions in both
directions.

Printing that was not enough. Three published dense 27B rungs shipped a
model.py containing `from mlx_lm.models.vq_switch import _dense_fused`, a
module that exists only in our development venvs. Two of them could not
generate a single token for anyone who downloaded them; the third fell back
to reconstructing every weight matrix per token at 0.43 tok/s. They passed
this smoke, because it ran where that module happened to exist. The tool was
testing the artifact in OUR environment, not in a downloader's.

So `--strict` (the default) now FAILS when the runtime resolves from anywhere
a downloader would not have it: a repo checkout, or a VQ-patched mlx-lm. Pass
--no-strict to get the old print-only behaviour.

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
    ap.add_argument("--strict", dest="strict", action="store_true",
                    default=True,
                    help="fail if the runtime resolves from anywhere a "
                         "downloader would not have it (default)")
    ap.add_argument("--no-strict", dest="strict", action="store_false",
                    help="print the resolved runtime without asserting on "
                         "it — the pre-2026-09-01 behaviour that let three "
                         "broken artifacts ship")
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
    def _class_ns(cls):
        """The namespace the class was actually defined in.

        mlx-lm loads a checkpoint's model.py with
        spec_from_file_location("custom_model", ...) + exec_module and never
        registers it in sys.modules, so `sys.modules[cls.__module__]` is None
        for every bundled artifact. Reading it that way made this check
        silently skip the exact case it exists for. The class's own functions
        close over the module dict, which does carry __file__.
        """
        mod = sys.modules.get(cls.__module__)
        if mod is not None and getattr(mod, "__file__", None):
            return vars(mod)
        for attr in vars(cls).values():
            g = getattr(attr, "__globals__", None)
            if g is not None and g.get("__file__"):
                return g
        return None

    seen, vq_modules, bundle_ns = set(), 0, None
    for name, mod in model.named_modules() if hasattr(model, "named_modules") \
            else []:
        cls = type(mod)
        if cls.__name__.startswith("VQ"):
            vq_modules += 1
            ns = _class_ns(cls)
            if bundle_ns is None:
                bundle_ns = ns
            path = (ns or {}).get("__file__")
            if path is None:
                import inspect
                try:
                    path = inspect.getfile(cls)
                except TypeError:
                    path = None
            key = (cls.__name__, path)
            if key not in seen:
                seen.add(key)
                print(f"  {cls.__name__:18s} <- {path or '<unknown origin>'}")
    if not seen:
        print("  (no VQ modules found — is this a VQ artifact?)")
    print(f"  {vq_modules} VQ module(s) instantiated")

    # ---------------------------------------------------------------- strict
    # A downloader has exactly two things: the artifact directory, and a
    # RELEASED mlx-lm. Anything the runtime resolves from outside those is a
    # copy they do not have, and every check that passed on it proved nothing
    # about their machine.
    #
    # "outside site-packages" is NOT the test, and getting that wrong is how
    # this shipped: our build venvs carry
    # site-packages/mlx_lm/models/vq_switch.py, which is inside site-packages
    # and still absent for every downloader. So the VQ runtime must come from
    # the ARTIFACT, and a patched mlx-lm is a failure in its own right.
    problems = []

    def _where(path):
        if not path or not str(path).startswith("/"):
            return "unknown"
        rp = str(pathlib.Path(path).resolve())
        if rp.startswith(str(art.resolve())):
            return "artifact"
        if "site-packages" in rp or "dist-packages" in rp:
            return "site-packages"
        return "checkout"

    # A. every VQ class must come from the artifact's own bundle.
    for cls_name, path in sorted(seen):
        w = _where(path)
        if w != "artifact":
            problems.append(
                f"{cls_name} resolved from {w} ({path}); a downloader has "
                f"only the artifact's model.py")

    # B. the fused kernel is resolved by NAME at call time, separately from
    #    the classes, and that indirection is exactly what broke. Ask the
    #    bundle to resolve it and check where the answer came from.
    # Families reach their kernels differently, and the gate has to check what
    # each one ACTUALLY does rather than impose one family's contract on all.
    #
    # Dense bundles resolve BY NAME at call time via _resolve_kernel -- that
    # indirection is precisely what shipped broken, so follow it. MoE bundles
    # call _fused straight out of module globals (vq_switch.py defines the
    # class and the kernel in the same file), so there is no indirection and
    # no _resolve_kernel. Requiring one of them was a false failure, and the
    # packer should not have to ship dense code it never uses just to satisfy
    # this check. A gate that cries wolf is a gate people pass --no-strict to.
    _rk = (bundle_ns or {}).get("_resolve_kernel")
    checked = 0
    for kname in ("_dense_fused", "_fused"):
        fn = None
        if _rk is not None:
            try:
                fn = _rk(kname)
            except Exception:
                fn = None
        if fn is None:
            fn = (bundle_ns or {}).get(kname)
        if fn is None:
            continue
        checked += 1
        fpath = getattr(fn, "__globals__", {}).get("__file__")
        w = _where(fpath)
        via = "_resolve_kernel" if _rk is not None else "module globals"
        print(f"  {kname:18s} <- {fpath} [{w}, via {via}]")
        if w != "artifact":
            problems.append(
                f"{kname} resolved from {w} ({fpath}); the bundle must "
                f"carry its own kernels")
    if bundle_ns is not None and checked == 0:
        problems.append(
            "the bundle exposes no fused kernel entry point (_fused or "
            "_dense_fused) by any route; its kernels cannot be traced to a "
            "source")

    # C. a VQ-patched mlx-lm masks exactly this class of defect.
    import importlib.util as _ilu
    try:
        spec = _ilu.find_spec("mlx_lm.models.vq_switch")
    except (ImportError, AttributeError, ValueError):
        spec = None
    if spec is not None:
        problems.append(
            f"mlx_lm.models.vq_switch exists in this environment "
            f"({getattr(spec, 'origin', '?')}). No released mlx-lm ships it, "
            f"so this venv cannot prove anything about a downloader's. "
            f"Remove it and re-run.")

    # D. mlx-lm itself must be an installed package, not a working copy.
    _mlx = sys.modules.get("mlx_lm")
    _mlxw = _where(getattr(_mlx, "__file__", None))
    if _mlx is not None and _mlxw == "checkout":
        problems.append(
            f"mlx_lm is a repo checkout ({_mlx.__file__}), not a released "
            f"install; a downloader pip-installs it")

    if problems:
        print("\nSTRICT: the runtime did not resolve the way a downloader's "
              "would:" if a.strict else
              "\nWARNING (--no-strict): a downloader would not resolve this "
              "runtime the same way:")
        for pr in problems:
            print(f"  - {pr}")
        if a.strict:
            print("\nFAIL: this environment cannot certify the artifact. "
                  "Re-run in a venv with a released mlx-lm and none of our "
                  "modules installed or on PYTHONPATH. (--no-strict to "
                  "override, but then the run proves nothing about a "
                  "downloader.)")
            return 1
    else:
        print("  STRICT OK: runtime resolved entirely from the artifact, "
              "against a released mlx-lm")

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
