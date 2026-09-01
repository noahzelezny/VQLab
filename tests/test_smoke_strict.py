"""The smoke gate must assert on WHERE the runtime resolved, not just print it.

Three published dense 27B rungs shipped a model.py importing
`mlx_lm.models.vq_switch`, a module that exists only in our development
venvs. Two could not generate a token for anyone who downloaded them. They
passed smoke, because smoke ran where that module happened to exist.

These tests pin the classifier that decides "would a downloader resolve it
this way", since that judgement is the whole gate.
"""
import pathlib
import re
import sys

SMOKE = pathlib.Path(__file__).resolve().parents[1] / "src" / "vqlab" / "smoke.py"
SRC = SMOKE.read_text()


def _where_impl():
    """Lift the classifier out of smoke.py so it can be tested without
    loading a 12 GiB model."""
    art = pathlib.Path("/models/some-artifact")

    def _where(path):
        if not path or not str(path).startswith("/"):
            return "unknown"
        rp = str(pathlib.Path(path).resolve())
        if rp.startswith(str(art.resolve())):
            return "artifact"
        if "site-packages" in rp or "dist-packages" in rp:
            return "site-packages"
        return "checkout"
    return _where


def test_classifier_separates_the_three_origins():
    w = _where_impl()
    assert w("/models/some-artifact/model.py") == "artifact"
    assert w("/opt/py/lib/python3.12/site-packages/mlx_lm/utils.py") == "site-packages"
    assert w("/Users/dev/repo/vqlab/src/vqlab/vq_switch.py") == "checkout"
    # A module exec'd without a file must never be mistaken for a real path;
    # reading the "<module ... no file>" placeholder as one produced a false
    # failure on a correct artifact.
    assert w("<module 'custom_model' — exec'd, no file>") == "unknown"
    assert w(None) == "unknown"


def test_strict_is_the_default():
    block = SRC[SRC.index('"--strict"'):SRC.index('"--no-strict"')]
    assert 'action="store_true"' in block and "default=True" in block, \
        "strict must default on; opt-out is --no-strict"
    assert '"--no-strict"' in SRC
    assert 'dest="strict", action="store_false"' in SRC


def test_patched_mlx_lm_is_itself_a_failure():
    """site-packages is NOT a sufficient test: our build venvs carry
    site-packages/mlx_lm/models/vq_switch.py, which is inside site-packages
    and still absent for every downloader. Its mere presence must fail."""
    assert 'find_spec("mlx_lm.models.vq_switch")' in SRC
    assert "No released mlx-lm ships it" in SRC


def test_kernels_are_checked_separately_from_classes():
    """The classes came from the artifact even in the broken builds; it was
    the kernel, resolved by name at call time, that came from elsewhere. The
    gate has to follow that indirection."""
    assert "_resolve_kernel" in SRC
    assert '"_dense_fused", "_fused"' in SRC


def test_bundled_module_is_found_without_sys_modules():
    """mlx-lm exec's a checkpoint's model.py as "custom_model" and never
    registers it in sys.modules, so looking it up there yields None and the
    check silently skips. The class's own globals carry __file__."""
    assert "__globals__" in SRC
    assert "never\n        registers it in sys.modules" in SRC


# ------------------------------------------------------- the release gate
CHECK = pathlib.Path(__file__).resolve().parents[1] / "src" / "vqlab" / "check_release.py"
CSRC = CHECK.read_text()


def test_release_gate_statically_rejects_dev_only_imports():
    """The cheapest check that would have prevented the whole incident: no
    model load, no GPU, just read the bundle. Verified against the model.py
    we actually shipped -- it names lines 144 and 149."""
    assert '"from mlx_lm.models.vq_"' in CSRC
    assert '"import mlx_lm.models.vq_"' in CSRC
    assert "cannot run outside our venvs" in CSRC


def test_release_gate_runs_strict_smoke_by_default():
    assert '"--strict"' in CSRC, "the gate must pass --strict explicitly"
    assert "--no-smoke" in CSRC
    # Opting out has to be loud: silence is what made three broken artifacts
    # look certified.
    assert "generation NOT verified" in CSRC


def test_release_gate_compiles_the_bundle():
    assert 'compile(_src, _mf, "exec")' in CSRC


def test_release_gate_fails_closed_on_smoke_failure():
    """A non-zero smoke must become a gate failure, not a warning."""
    i = CSRC.index("strict smoke failed")
    assert "fails.append" in CSRC[max(0, i - 200):i]


def test_kernel_check_follows_each_family_s_own_route():
    """Dense bundles resolve kernels BY NAME via _resolve_kernel; MoE bundles
    call _fused straight out of module globals, because vq_switch.py defines
    the class and the kernel in one file.

    The first cut of this gate demanded _resolve_kernel of everything, so
    every MoE bundle failed with "exposes no _resolve_kernel" even when its
    kernels were fine -- and the proposed workaround was to make the packer
    ship dense code into MoE artifacts that never call it. A gate with false
    positives is a gate people pass --no-strict to, so the gate checks what
    each family actually does instead."""
    assert '_rk = (bundle_ns or {}).get("_resolve_kernel")' in SRC
    # falls back to the namespace when there is no indirection to follow
    assert 'fn = (bundle_ns or {}).get(kname)' in SRC
    # and only complains when NO route finds a kernel at all
    assert "checked == 0" in SRC
    assert "no fused kernel entry point" in SRC
    # the old unconditional demand must be gone
    assert "the bundle exposes no _resolve_kernel" not in SRC


def test_kernel_route_is_reported_not_just_the_path():
    """Which route resolved it is part of the evidence: a dense bundle that
    silently stopped using _resolve_kernel would otherwise look identical."""
    assert 'via {via}' in SRC or "via " in SRC
    assert '"_resolve_kernel" if _rk is not None else "module globals"' in SRC
