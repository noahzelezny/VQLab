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

import re

FLAGS = ("VQ_GEMMSEG_BF16IO", "VQ_DECODE_BF16IO")

# Every env-var default baked into a bundled runtime, profile flags included.
# A REBUNDLE must reproduce the values the artifact shipped with, because the
# repo's current defaults are not the artifact's: they move whenever an arc
# lands a new default, and the rebundle is usually being run for an unrelated
# reason. Both halves of that were live on 2026-09-19 -- rebundling the gemma
# rungs to repair their vision tower ALSO flipped VQ_DENSE_SS 0->1 (F148's
# default, whose measured scope is the 27B dense rungs ONLY) and, at the
# default --runtime, would have downgraded a v2 artifact to v1.5. Neither
# change was asked for, and check-bundle accepts EITHER bf16 profile, so
# neither would have been caught.
#
# Anything matching `os.environ.get("VQ_...", "<default>")` is tracked. A new
# flag is picked up automatically; it does not need registering here.
_ENV_DEFAULT = re.compile(
    r'os\.environ\.get\(\s*"(VQ_[A-Z0-9_]+)"\s*,\s*"([^"]*)"\s*\)')


def flags_of(text):
    """{flag: baked default} for every VQ_* env default in `text`. First
    occurrence wins, which is how the runtime itself reads them."""
    out = {}
    for flag, default in _ENV_DEFAULT.findall(text):
        out.setdefault(flag, default)
    return out


def apply_flags(text, wanted):
    """Return `text` with each flag in `wanted` rewritten to that default.
    Flags absent from `text` are ignored -- a runtime that no longer has a
    knob is not made to grow one back."""
    def sub(m):
        flag = m.group(1)
        if flag not in wanted:
            return m.group(0)
        return f'os.environ.get("{flag}", "{wanted[flag]}")'
    return _ENV_DEFAULT.sub(sub, text)


def preserve_shipped(shipped_text, new_text, override=()):
    """Carry the SHIPPED runtime's baked defaults into `new_text`.

    Returns (text, carried, changed, dropped):
      carried -- flags pinned back to their shipped value
      changed -- [(flag, shipped, repo)] the repo would have altered, so the
                 caller can PRINT them; silence is how the ride-along shipped
      dropped -- flags in `override`, left at the repo's value deliberately
    """
    shipped, repo = flags_of(shipped_text), flags_of(new_text)
    want, changed, dropped = {}, [], []
    for flag, was in shipped.items():
        if flag not in repo:
            continue
        if flag in override:
            if repo[flag] != was:
                dropped.append((flag, was, repo[flag]))
            continue
        if repo[flag] != was:
            changed.append((flag, was, repo[flag]))
        want[flag] = was
    return apply_flags(new_text, want), want, changed, dropped


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


def resolve_runtime(shipped_text, new_text, profile=None, adopt=()):
    """The one place a bundler decides which runtime defaults it writes.

    `shipped_text` is the artifact's existing model.py (None on a fresh
    build). `profile` is an EXPLICIT --runtime, or None to leave the bf16
    pair wherever preservation puts it. `adopt` names flags that should
    deliberately take the repo's current default instead of the shipped one
    ("all" adopts every flag, i.e. the old behaviour).

    Returns (text, report) where report is printable lines. A rebundle that
    changes a shipped default and does not SAY SO is the failure mode this
    exists to prevent, so the report is never empty when something moved.
    """
    report = []
    if shipped_text is None:
        text = apply_profile(new_text, profile or "v1.5")
        report.append(f"fresh build at runtime profile {profile or 'v1.5'}")
        return text, report

    adopt_all = "all" in adopt
    override = tuple(flags_of(new_text)) if adopt_all else tuple(adopt)
    text, carried, changed, dropped = preserve_shipped(
        shipped_text, new_text, override=override)

    if changed:
        report.append(
            f"preserved {len(changed)} shipped runtime default(s) the repo "
            f"would have changed (--adopt to take the repo's):")
        for flag, was, repo in sorted(changed):
            report.append(f"    {flag:26s} shipped={was!r}  repo={repo!r} -> kept {was!r}")
    if dropped:
        report.append(f"ADOPTED the repo's default for {len(dropped)} flag(s), as asked:")
        for flag, was, repo in sorted(dropped):
            report.append(f"    {flag:26s} shipped={was!r} -> {repo!r}")
    if profile:
        before = profile_of(text)
        text = apply_profile(text, profile)
        report.append(f"runtime profile forced to {profile}"
                      + (f" (bundle carried {before})" if before != profile else ""))
    else:
        report.append(f"runtime profile preserved: {profile_of(text)}")
    if not changed and not dropped:
        report.insert(0, f"runtime defaults unchanged from shipped "
                         f"({len(carried)} flags checked)")
    return text, report
