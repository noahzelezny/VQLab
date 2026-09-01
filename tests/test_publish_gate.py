"""`vqlab publish` must be the only sanctioned upload path, and must not be
bypassable.

On 2026-09-01 three dense 27B rungs were uploaded with a bundled model.py
importing a module that exists only in our development venvs; two of them
could not generate a token for anyone who downloaded them. `check-release`
would have caught it. Nothing made `check-release` run.

These tests pin the properties that make forgetting impossible rather than
merely discouraged.
"""
import pathlib

SRC = (pathlib.Path(__file__).resolve().parents[1] / "src" / "vqlab"
       / "publish.py").read_text()


def test_gate_runs_before_any_upload():
    """The gate call must precede the import of the upload client, so no
    reordering can quietly upload first."""
    assert SRC.index("check_release.py") < SRC.index("from huggingface_hub")


def test_gate_failure_returns_before_uploading():
    i = SRC.index("REFUSING TO UPLOAD: the release gate failed")
    assert "return 1" in SRC[i:i + 400]


def test_there_is_no_override():
    """A bypass that exists is a bypass that gets used at 2am."""
    # Scoped to declared arguments: the docstring mentions these flags
    # precisely to say they do not exist, which is not the same thing.
    declared = [ln for ln in SRC.splitlines() if "add_argument(" in ln]
    blob = "\n".join(declared)
    for flag in ("--force", "--skip-gate", "--no-gate", "--no-check",
                 "--yes-i-know", "--allow-unverified"):
        assert flag not in blob, f"{flag} must not be a declared argument"
    assert "no --force and no --skip-gate" in SRC


def test_files_are_hashed_before_and_after_the_gate():
    """The gate certifies bytes at the moment it runs. A bundled model.py is
    generated from a working tree, and another session can be editing that
    tree -- which nearly happened while this was written."""
    assert "before = {p: _digest(p) for p in targets}" in SRC
    assert "_digest(p) != before[p]" in SRC
    i = SRC.index("changed while the gate was")
    assert "return 1" in SRC[i:i + 400]


def test_containment_is_lexical_not_resolved():
    """resolve() follows symlinks, so a legitimately symlinked artifact file
    looks like an escape; normpath settles the '../..' case without that."""
    assert "os.path.normpath" in SRC
    assert "points outside the artifact directory" in SRC
    assert ".resolve()\n            if not p.is_file()" not in SRC


def test_dry_run_uploads_nothing():
    i = SRC.index("if a.dry_run:")
    assert "return 0" in SRC[i:i + 200]
    assert SRC.index("if a.dry_run:") < SRC.index("from huggingface_hub")


def test_docs_only_uploads_skip_the_smoke_but_not_the_static_checks():
    """A README cannot break the runtime, and the smoke needs the whole model
    resident -- 112 GiB for the largest Flash-Next rung. Demanding that to fix
    a sentence produces a gate people route around, which is the failure the
    gate exists to prevent. So documentation-only uploads run the static scan
    and skip the generation smoke, and say so in the output.

    Anything touching the runtime or config still takes the full gate."""
    assert 'DOC_SUFFIXES = {".md", ".txt", ".jinja"}' in SRC
    assert "docs_only = all(" in SRC
    # the static half is NOT skipped -- --no-smoke keeps check_release's
    # import scan and compile check
    assert 'gate.append("--no-smoke")' in SRC
    # check_release is invoked on BOTH paths -- the gate command is built
    # once and only gains --no-smoke for docs.
    assert SRC.count("check_release.py") >= 1
    i = SRC.index("gate = [sys.executable")
    j = SRC.index("r = subprocess.run(gate)")
    assert "check_release.py" in SRC[i:j]
    assert SRC.index("r = subprocess.run(gate)") < SRC.index("from huggingface_hub")
    # and the user is told which verification actually happened
    assert "generation NOT re-verified" in SRC
