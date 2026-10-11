#!/usr/bin/env python3
"""Gate: an artifact's bundled model.py must carry the CURRENT repo runtime.

Three verdicts: PASS (current runtime), STALE (an earlier runtime certified
byte-identical in output -- runtime/equivalent_revisions.json -- so the bundle
is only missing speedups; exit 0, or 2 with --strict), FAIL (anything else).

External users run the bundle; benches run the venv runtime. Any drift means
published speed/quality claims describe code downloaders don't have.

MoE artifacts (config declares vq_modules): PASS = repo vq_switch.py text is
contained verbatim in the bundle.

Dense artifacts (config declares vq_linear/vq_embed): the runtime is
vq_dense.py, whose fused path needs _dense_fused/_fused from vq_switch.py,
so a correct dense bundle carries BOTH files' text (build_dense.py
writes them in that order, followed by the loader shim). Dense bundles that
carry vq_dense.py ALONE reach into `mlx_lm.models.vq_switch` at call time
and therefore require a VQ-patched mlx-lm: measured, such an artifact
raises ModuleNotFoundError on a stock install — it scores fine and cannot
serve. Passing this gate is still not proof of which copy executes; see
METHODOLOGY.md §5 and run bundle-accept.

ABSENT vs DRIFTED (2026-09-09). Verbatim containment answers "is this the
CURRENT text", not "is the runtime here at all", and the dense branch used
to report both as the ModuleNotFoundError case. It was wrong on all three
27B rungs and gemma-e4b-PLE: each carries vq_switch inline (VQSwitchLinear,
_dense_fused, _fused and _resolve_kernel are all DEFINED in the bundle) and
differs from the repo only by the 62 lines of RTILE/CB_DEV work that landed
after they were spliced. That is ordinary drift, the same failure the other
15 artifacts have — not a stock-install break. So absence is now decided by
whether the runtime's top-level defs are defined in the bundle, and only
that case claims ModuleNotFoundError.
"""
import argparse
import json
import pathlib
import re
import sys


# The symbols a dense bundle's fused path actually reaches for. If these are
# DEFINED in model.py, _resolve_kernel finds them in its own globals (tier 1)
# and never touches mlx_lm.models.vq_switch -- so the artifact loads on a
# stock install regardless of how far the rest of the text has drifted.
# Do NOT widen this to "every top-level def": a kernel added to the repo
# AFTER an artifact was spliced (gemmseg_cb_dev, 2026-09-08) is absent from
# every older bundle, and reporting that as ModuleNotFoundError is exactly
# the false alarm this function exists to end. Missing-and-uncalled is drift.
_DENSE_ANCHORS = {
    "vq_switch.py": ("VQSwitchLinear", "_dense_fused", "_fused"),
    "vq_dense.py": ("VQLinear", "VQEmbedding", "_decode_matmul",
                    "_resolve_kernel"),
}


def _undefined_anchors(name: str, bundle: str) -> list:
    """Load-bearing symbols of `name` that `bundle` never defines."""
    return [a for a in _DENSE_ANCHORS.get(name, ())
            if not re.search(rf"^(?:def|class)\s+{re.escape(a)}\b", bundle, re.M)]


# What a registry entry without `flags` was certified at (and what every
# revision older than the knobs bakes, since an absent knob runs these).
LEGACY_FLAGS = {"VQ_FUSED_MAX_N": 4096, "VQ_GEMMSEG_ACCS": False, "VQ_GEMMSEG_CVEC": False}


def _baked_prefill_flags(bundle: str, rp) -> dict:
    """The bundle's baked values of the certification flags, typed like the
    registry (int / bool); a knob the text does not have runs its legacy value."""
    have = rp.flags_of(bundle)
    out = {}
    for k, legacy in LEGACY_FLAGS.items():
        v = have.get(k)
        if v is None:
            out[k] = legacy
        elif isinstance(legacy, bool):
            out[k] = v.strip().lower() not in ("", "0", "false", "off", "no")
        else:
            try:
                out[k] = int(v)
            except ValueError:
                out[k] = v
    return out


def _certified_flags(rev) -> dict:
    return {k: type(LEGACY_FLAGS[k])(v)
            for k, v in dict(LEGACY_FLAGS, **rev.get("flags", {})).items()}


def _stale_as(bundle: str, name: str, rp):
    """(rev, near) for runtime file `name` (runtime/equivalent_revisions.json).

    rev is the certified-equivalent revision `bundle` carries AT THE FLAGS IT
    WAS CERTIFIED AT (its `flags`, legacy 4096/off/off when absent), else
    None. Certified = its OUTPUT is byte-identical to the current runtime's
    for every artifact without vq_skipzero (tests/test_runtime_equivalence.py),
    so a bundle on it is STALE -- slower, same numbers -- not broken.

    The CODE match is modulo VQ_* defaults (a rebundle preserves the shipped
    ones), so the flags are compared separately: the certification holds at
    one flag set, and a bundle baking others is not covered by it. When the
    code matches a revision whose certified flags differ from the bundle's
    baked ones, near lists [(revision sha, {flag: (bundle, certified)})]."""
    from vqlab._layout import runtime_file
    reg_path = runtime_file("equivalent_revisions.json")
    reg = json.loads(reg_path.read_text())
    baked = _baked_prefill_flags(bundle, rp)
    near = []
    for rev in reg.get("revisions", []):
        if pathlib.PurePath(rev["path"]).name != name:
            continue
        old = (reg_path.parent / rev["file"]).read_text()
        if not (rp.matches_any_profile(bundle, old)[0] or rp.matches_modulo_flags(bundle, old)[0]):
            continue
        cert = _certified_flags(rev)
        # the certification flags are vq_switch's prefill knobs; a vq_dense
        # revision is certified independent of them
        deltas = {k: (baked[k], cert[k]) for k in cert
                  if name == "vq_switch.py" and baked[k] != cert[k]}
        if not deltas:
            return rev, []
        near.append((rev["sha256"], deltas))
    return None, near


def check(artifact, strict=False, stale_revs=None):
    """(exit code, report lines) for one artifact's bundle: 0 PASS (or STALE
    without strict), 1 FAIL, 2 STALE under strict. main() prints the lines;
    vqlab.api.gate returns them. On STALE, the certified revisions the bundle
    carries are appended to `stale_revs` when one is given."""
    lines = []
    _out = lines.append
    from vqlab._layout import runtime_file
    art = pathlib.Path(artifact)

    cfg = {}
    cfg_path = art / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
    dense = bool(cfg.get("vq_linear") or cfg.get("vq_embed"))

    bp = art / "model.py"
    if not bp.exists():
        _out("FAIL: no bundled model.py")
        return 1, lines
    bundle = bp.read_text()

    if dense:
        vd = (runtime_file("vq_dense.py")).read_text()
        vs = (runtime_file("vq_switch.py")).read_text()
        absent, drifted = [], []
        for name, text in (("vq_dense.py", vd), ("vq_switch.py", vs)):
            if text in bundle:
                continue
            undefined = _undefined_anchors(name, bundle)
            (absent if undefined else drifted).append((name, undefined))
        if absent:
            detail = "; ".join(
                f"{n} (undefined: {', '.join(u)})" for n, u in absent)
            _out(f"FAIL (dense artifact): bundle is missing {detail}. "
                  "Its fused path would resolve against site-packages, so the "
                  "artifact requires a VQ-patched mlx-lm and raises "
                  "ModuleNotFoundError on a stock install. Re-run build-dense "
                  "to write a bundle carrying both runtimes.")
            return 1, lines
        if drifted:
            # Same code, different baked defaults is not drift -- see the
            # dense-equivalent note on the MoE path below.
            import importlib.util as _ilu0
            _sp = _ilu0.spec_from_file_location(
                "runtime_profile", runtime_file("runtime_profile.py"))
            _rp = _ilu0.module_from_spec(_sp)
            _sp.loader.exec_module(_rp)
            oks, all_deltas = [], {}
            for name, text in (("vq_dense.py", vd), ("vq_switch.py", vs)):
                o, d = _rp.matches_modulo_flags(bundle, text)
                oks.append(o)
                all_deltas.update(d)
            if all(oks):
                _out(f"PASS (dense artifact): bundle carries both runtimes "
                      f"verbatim, with {len(all_deltas)} baked default(s) "
                      f"differing from the repo:")
                for flag, (have, repo) in sorted(all_deltas.items()):
                    _out(f"    {flag:26s} bundle={have!r}  repo={repo!r}")
                _out("  These are the values the artifact SHIPPED with and "
                      "were preserved deliberately. Any runtime claim must "
                      "name them. (Still instrument the resolved import "
                      "before any runtime claim.)")
                return 0, lines
            stale = {}
            for name, _u in drifted:
                rev, near = _stale_as(bundle, name, _rp)
                if rev is None:
                    for sha, deltas in near:
                        _out(f"not STALE: {name} matches certified runtime @ {sha[:12]}, "
                              f"but the bundle bakes defaults that certification does "
                              f"not cover:")
                        for flag, (have, cert) in sorted(deltas.items()):
                            _out(f"    {flag:26s} bundle={have!r}  certified={cert!r}")
                    break
                stale[name] = rev
            else:
                if stale_revs is not None:
                    stale_revs.extend(stale.values())
                _out("STALE (dense artifact): the bundle carries an earlier runtime "
                      "whose output is byte-identical to the current one; rebundle "
                      "to pick up:")
                for name, rev in stale.items():
                    _out(f"    {name} @ {rev['sha256'][:12]}: {rev['gains']}")
                return (2 if strict else 0), lines
            names = ", ".join(n for n, _ in drifted)
            _out(f"FAIL (dense artifact): bundled model.py "
                  f"({len(bundle.splitlines())} lines) carries every top-level "
                  f"def of {names}, but not the CURRENT text. This is drift, "
                  "not a missing runtime \u2014 the artifact loads on a stock "
                  "install and runs code older than the benches. Re-splice "
                  "before publishing any claim.")
            return 1, lines
        _out("PASS: dense bundle carries both runtimes verbatim (still "
              "instrument the resolved import before any runtime claim)")
        return 0, lines

    # SKIPZERO (docs/SKIPZERO.md): the compact rows are only safe under a
    # bundle whose loader builds row-table modules. The experimental stage-1/2
    # hooks append their own loaders and are never a shipped runtime.
    sz = cfg.get("vq_skipzero")
    if sz:
        bad = [h for h in ("skipzero_load", "sz_resident") if h in bundle]
        if bad or sz.get("loader") != "runtime" or "_SZ_MODS" not in bundle:
            _out(f"FAIL: config declares vq_skipzero but the bundle does not "
                  f"serve it through the runtime (loader={sz.get('loader')!r}, "
                  f"experimental hooks present: {bad or 'none'}). Re-run "
                  f"`vqlab bundle` on it.")
            return 1, lines
        missing = [m for m in sz.get("modules", {})
                   if m not in cfg.get("vq_modules", {})]
        if missing:
            _out(f"FAIL: {len(missing)} skipzero module(s) missing from "
                  f"vq_modules, e.g. {missing[0]}")
            return 1, lines
        # The runtime serves compact rows only at the dims skipzero_load
        # declares; any other dim crashes on the first prefill, not at load.
        import importlib.util as _ilu
        _p = pathlib.Path(__file__).resolve().parents[1] / "skipzero" / \
            "skipzero_load.py"
        _spec = _ilu.spec_from_file_location("skipzero_load", _p)
        _szl = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_szl)
        vqm = cfg.get("vq_modules", {})
        wrong = [m for m in sz.get("modules", {})
                 if not _szl.servable(vqm[m])]
        if wrong:
            _out(f"FAIL: {len(wrong)} skipzero module(s) have a codebook dim "
                  f"the runtime has no skipzero kernel for, e.g. {wrong[0]} "
                  f"(dim={vqm[wrong[0]].get('dim')}; supported: "
                  f"{list(_szl.SUPPORTED_DIMS)}, d{_szl.PACKED_ONLY_DIMS} "
                  f"packed only). Re-run sz-pack.")
            return 1, lines
        _out(f"skipzero: {len(sz.get('modules', {}))} module(s) served "
              f"resident through the runtime row-table switch")

    runtime = (runtime_file("vq_switch.py")).read_text()
    # Two legitimate runtime PROFILES differ by exactly the two bf16-I/O flag
    # defaults (docs/RUNTIME-SHIP-PLAN.md). Verify against either, and SAY
    # WHICH -- otherwise the repo's current default silently decides which
    # artifacts can pass their own gate, and flipping it to publish one
    # artifact breaks every other.
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("runtime_profile",
                                         runtime_file("runtime_profile.py"))
    runtime_profile = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(runtime_profile)
    ok, profile = runtime_profile.matches_any_profile(bundle, runtime)
    if ok:
        _out(f"PASS: bundle carries the current runtime "
              f"({len(runtime.splitlines())} lines) verbatim "
              f"[profile: {profile}]")
        return 0, lines
    # Same CODE, different baked DEFAULTS is not drift. A rebundle preserves
    # the defaults the artifact shipped with (runtime_profile.resolve_runtime),
    # so any tracked VQ_* flag can legitimately differ from the repo's current
    # value -- not just the bf16 pair the profile check knows about. Report it
    # in full rather than failing: "runs the current code with these defaults"
    # is a different claim from "runs the current code", and a reader of this
    # gate needs to see which flags, not just that it passed.
    ok, deltas = runtime_profile.matches_modulo_flags(bundle, runtime)
    if ok:
        _out(f"PASS: bundle carries the current runtime "
              f"({len(runtime.splitlines())} lines) verbatim, with "
              f"{len(deltas)} baked default(s) differing from the repo:")
        for flag, (have, repo) in sorted(deltas.items()):
            _out(f"    {flag:26s} bundle={have!r}  repo={repo!r}")
        _out("  These are the values the artifact SHIPPED with and were "
              "preserved deliberately. Any runtime claim must name them.")
        return 0, lines
    rev, near = (None, []) if cfg.get("vq_skipzero") else \
        _stale_as(bundle, "vq_switch.py", runtime_profile)
    if rev is not None:
        if stale_revs is not None:
            stale_revs.append(rev)
        fl = ", ".join(f"{k}={int(v)}" for k, v in _certified_flags(rev).items())
        if rev.get("flags"):
            # Certified at its OWN baked prefill defaults, which this bundle
            # also bakes (checked in _stale_as).
            _out(f"STALE: bundle carries runtime @ {rev['sha256'][:12]} baking {fl}, whose "
                  f"output at those defaults is byte-identical to the current runtime's "
                  f"(tests/test_runtime_equivalence.py). A rebundle adds: {rev['gains']}")
            return (2 if strict else 0), lines
        # Legacy entries are certified at the LEGACY prefill defaults
        # (VQ_FUSED_MAX_N=4096, ACCS/CVEC off), which this bundle bakes: since
        # the F200-F204 flip, the current runtime takes gemmseg2+ACCS for
        # 512 < N <= 4096 pairs, so "byte-identical to the current runtime" is
        # no longer true for those prompts. Still STALE (not broken), but say so.
        _out(f"STALE: bundle carries runtime @ {rev['sha256'][:12]} baking the LEGACY "
              f"prefill defaults ({fl}); its output is byte-identical to the current "
              f"runtime run at those defaults (tests/test_runtime_equivalence.py). At the "
              f"current defaults, prompts of 512..4096 (token, expert) pairs differ: a "
              f"rebundle adds the faster, ACCS-corrected short-prompt prefill (F200-F204) "
              f"and {rev['gains']}")
        return (2 if strict else 0), lines
    for sha, deltas in near:
        _out(f"not STALE: the code matches certified runtime @ {sha[:12]}, but the bundle "
              f"bakes defaults that certification does not cover:")
        for flag, (have, cert) in sorted(deltas.items()):
            _out(f"    {flag:26s} bundle={have!r}  certified={cert!r}")
    _out(f"FAIL: bundled model.py ({len(bundle.splitlines())} lines) does not "
          f"contain the current runtime ({len(runtime.splitlines())} lines). "
          "Downloaders run different code than the benches. Re-splice before "
          "publishing any claim.")
    return 1, lines



_VERDICTS: dict = {}   # artifact + runtime file mtimes -> verdict
_RUNTIME_INPUTS = ("vq_switch.py", "vq_dense.py", "equivalent_revisions.json")


def verdict(artifact) -> dict:
    """check-bundle in one word, for the GUI: PASS (current runtime), CURRENT
    (an earlier certified revision whose gains apply only to features this
    artifact does not use, so a rebundle changes nothing), STALE (a rebundle
    is worth doing), FAIL, or MISSING (no model.py). `note` carries the
    revisions' `gains`, or why it failed. Never raises; cached until the
    artifact's model.py / config.json or the runtime it is compared with
    changes."""
    from vqlab._layout import runtime_file
    art = pathlib.Path(artifact)
    try:
        key = (str(art), (art / "model.py").stat().st_mtime_ns,
               (art / "config.json").stat().st_mtime_ns,
               *(runtime_file(n).stat().st_mtime_ns for n in _RUNTIME_INPUTS))
    except OSError:
        return {"verdict": "MISSING", "note": "no model.py or config.json"}
    if key not in _VERDICTS:
        _VERDICTS[key] = _verdict(art)
    return _VERDICTS[key]


def _verdict(art: pathlib.Path) -> dict:
    revs = []
    try:
        rc, lines = check(art, strict=True, stale_revs=revs)
        word = {0: "PASS", 2: "STALE"}.get(rc, "FAIL")
        if word == "FAIL":
            return {"verdict": word, "note": next((l for l in lines if "FAIL" in l), "")}
        note = "; ".join(r.get("gains", "") for r in revs)
        if word == "STALE":
            cfg = json.loads((art / "config.json").read_text())
            if revs and all(r.get("gains_only_for") is not None
                            and not any(cfg.get(k) for k in r["gains_only_for"]) for r in revs):
                word = "CURRENT"
        return {"verdict": word, "note": note}
    except Exception as e:  # one bad artifact is one FAIL row, never a failed page
        return {"verdict": "FAIL", "note": f"check-bundle could not run: {type(e).__name__}: {e}"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--strict", action="store_true",
                    help="exit 2 on STALE (default: STALE exits 0; it is a to-do, not a defect)")
    a = ap.parse_args(argv)
    rc, lines = check(a.artifact, a.strict)
    for line in lines:
        print(line)
    return rc

if __name__ == "__main__":
    sys.exit(main())
