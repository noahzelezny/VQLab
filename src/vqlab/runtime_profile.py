"""Runtime PROFILES for a bundled model.py, and how the gate accepts them.

An artifact's bundled `model.py` is `vq_switch.py` spliced in verbatim, and
`check-bundle` enforces that verbatim match -- which is what stops a bundle
drifting from the runtime the benches were run against. But two legitimate
profiles exist (docs/RUNTIME-SHIP-PLAN.md):

  v1.5  both bf16-I/O flags default OFF -- bit-exact with the published arc6
        runtime. What a repo shipping UNCHANGED weights gets.
  v2    both ON -- faster, at a small family-local accuracy cost. What a repo
        whose own quality gain pays for that cost may ship.

They differ by exactly two characters, so the gate can verify a bundle against
either profile instead of against one hard-coded default. Without this, the
repo's current default silently decides which artifacts can pass their own
gate, and flipping it to publish one artifact breaks every other.
"""

FLAGS = ("VQ_GEMMSEG_BF16IO", "VQ_DECODE_BF16IO")


def _line(flag, default):
    var = "_GEMMSEG_BF16IO" if "GEMMSEG" in flag else "_DECODE_BF16IO"
    return f'{var} = os.environ.get("{flag}", "{default}") == "1"'


def apply_profile(runtime_text, profile):
    """Return `runtime_text` with the two bf16-I/O defaults set for `profile`."""
    if profile not in ("v1.5", "v2"):
        raise ValueError(f"unknown runtime profile {profile!r}")
    want = "1" if profile == "v2" else "0"
    out = runtime_text
    for flag in FLAGS:
        for have in ("0", "1"):
            out = out.replace(_line(flag, have), _line(flag, want))
    return out


def profile_of(runtime_text):
    """Which profile a runtime text carries, or None if it is mixed."""
    on = [_line(f, "1") in runtime_text for f in FLAGS]
    if all(on):
        return "v2"
    if not any(on):
        return "v1.5"
    return None


def matches_any_profile(bundle_text, runtime_text):
    """(ok, profile) -- does `bundle_text` carry `runtime_text` under EITHER
    profile? Returns the profile it matched so the gate can report it."""
    for prof in ("v1.5", "v2"):
        if apply_profile(runtime_text, prof) in bundle_text:
            return True, prof
    return False, None
