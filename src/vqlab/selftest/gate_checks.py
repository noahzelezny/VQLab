"""Selftest: the gates (outlier verify, check-bundle, pin, step verdict,
spelling)."""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil

import mlx.core as mx

from vqlab._layout import find as _find
from vqlab.selftest import check, run
from vqlab.util.fs import read_json, write_json
from vqlab.util.hashing import sha256_file


def check_outlier_gate(ctx):
    print("[2/7] outlier gate")
    tmp, v = ctx.tmp, ctx.verbose
    p = run([str(_find("verify_artifact.py")), "--artifact", str(ctx.f1),
             "--src", str(ctx.src), "--family", "qwen3_8_dense",
             "--outlier", "3.0"], verbose=v)
    check("gate PASSES a healthy artifact", p.returncode == 0)

    bad = tmp / "fit-corrupt"
    shutil.copytree(ctx.f1, bad)
    sh = bad / "model-00001-of-00001.safetensors"
    # Read EAGERLY on the cpu stream before writing the same path back.
    # A lazy mx.load left unevaluated is materialized inside the save's
    # command buffer — i.e. read from a file already being overwritten,
    # which scrambles every tensor, not just the one being corrupted
    # (FINDINGS IV.1; hit while writing this very test).
    with mx.stream(mx.cpu):
        w = dict(mx.load(str(sh)))
        mx.eval(list(w.values()))
    w[ctx.m0 + ".codebook"] = mx.zeros_like(w[ctx.m0 + ".codebook"])  # collapse one
    mx.save_safetensors(str(sh), w)
    p = run([str(_find("verify_artifact.py")), "--artifact", str(bad),
             "--src", str(ctx.src), "--family", "qwen3_8_dense",
             "--outlier", "3.0"], expect_rc=1, verbose=v)
    check("gate FAILS a collapsed tensor (known-bad)", p.returncode != 0)


def check_bundle_gate(ctx):
    print("[5/7] bundle gate")
    for nm, files, dense, want in (
            ("moe-good", ["vq_switch.py"], False, 0),
            ("moe-stale", [], False, 1),
            ("dense-good", ["vq_switch.py", "vq_dense.py"], True, 0),
            ("dense-missing-switch", ["vq_dense.py"], True, 1)):
        d = ctx.tmp / ("bundle-" + nm)
        d.mkdir()
        write_json(d / "config.json", {"vq_linear" if dense else "vq_modules": {"x": {}}})
        (d / "model.py").write_text(
            "".join((_find(f)).read_text() for f in files) or "# stale\n")
        p = run([str(_find("check_bundle.py")), "--artifact", str(d)],
                expect_rc=want, verbose=ctx.verbose)
        check(f"check-bundle {nm} -> {'PASS' if want == 0 else 'FAIL'}",
              p.returncode == want)


def check_pin_and_step_verdict(ctx):
    print("[6a3/7] pin + step verdict: scorers refuse what was not smoked")
    _check_pin(ctx)
    _check_step_verdict(ctx)


def _check_pin(ctx):
    from vqlab.gate import pin as P
    from vqlab.records.provenance import measured as _meas2
    from vqlab.selftest.synthetic_model import make_source
    tmp = ctx.tmp
    ctx.psrc = psrc = tmp / "pin_src"
    make_source(psrc)
    (psrc / "model.py").write_text("# fixture runtime\n")
    pout = tmp / "pin_out"
    with contextlib.redirect_stdout(io.StringIO()):
        prc = P.pin(psrc, pout, headroom=1e-12)   # forces the RAM refusal
    prec = read_json(pout / P.MARKER)
    check("pin: weights symlinked to the resolved source, metadata copied",
          (pout / "model-00001-of-00001.safetensors").is_symlink()
          and not (pout / "config.json").is_symlink())
    check("pin: too big for the box -> deferred-ram, allowed with a WARNING",
          prc == 0 and prec["state"] == "deferred-ram"
          and P.check_pin(pout)[0] and "WARNING" in P.check_pin(pout)[1])
    try:
        P.pin(psrc, pout)
        refused_overwrite = False
    except SystemExit:
        refused_overwrite = True
    check("pin: never overwrites an existing directory", refused_overwrite)
    (pout / "model.py").write_text("# rebundled after the smoke\n")
    check("pin: model.py changed after pinning -> refused", not P.check_pin(pout)[0])
    prec["state"], prec["model_py_sha256"] = "failed", sha256_file(pout / "model.py")
    (pout / P.MARKER).write_text(json.dumps(prec))
    check("pin: failed smoke -> refused", not P.check_pin(pout)[0])
    check("pin: an unpinned directory is allowed unchanged", P.check_pin(psrc) == (True, ""))
    mp_ = _meas2(pout)
    check("a pin's stamp names its source and state, and shares its source's bytes",
          os.path.realpath(mp_["pin"]["source"]) == os.path.realpath(psrc)
          and mp_["pin"]["state"] == "failed"
          and mp_["fingerprint"] == _meas2(psrc)["fingerprint"])


def _check_step_verdict(ctx):
    from vqlab.records.step_verdict import step_verdict
    so, se = ctx.tmp / "step.out", ctx.tmp / "step.err"
    so.write_text("")
    se.write_text("Traceback (most recent call last):\n  File x\nFileNotFoundError: corpus\n")
    v = step_verdict(0, so, se, {"stdout_regex": "^{", "min_lines": 1})
    check("step_verdict: rc 0 + traceback + no output is a FAIL (the night-4 speed bug)",
          not v["ok"] and "FileNotFoundError" in v["stderr_tail"][-1])
    so.write_text('{"arm": "a"}\n')
    se.write_text("")
    check("step_verdict: rc 0 + expected output passes",
          step_verdict(0, so, se, {"stdout_regex": "^{"})["ok"])
    check("step_verdict: nonzero exit fails", not step_verdict(3, so, se)["ok"])


def check_spelling(ctx):
    from vqlab.gate import spelling as SP
    sp_in = "The `labelled` path is labelled; layer-wise noise, precise analyses, quantisation."
    check("spelling: flags British prose, spares code spans and US words",
          [w for _, w, _ in SP.scan(sp_in)] == ["labelled", "quantisation"])
    check("spelling: --fix rewrites to US and preserves code spans",
          SP.fix(sp_in) == "The `labelled` path is labeled; layer-wise noise, precise analyses, quantization.")
