"""vqlab.api -- the stable public surface. Read this one file to see what
vqlab can do; everything else in the package is internal and may move.

Built for MCP servers and library callers:

* every function is keyword-only and returns a dataclass from this module or
  a dict of JSON types, so an MCP tool is ``return asdict(api.X(**params))``
  (``to_json(x)`` does that for either kind);
* nothing here prints, calls ``sys.exit`` or reads ``sys.argv``; a refusal
  or failure raises :class:`ApiError` with a stable ``code``;
* long GPU jobs never block: they return a :class:`RunHandle` from the SAME
  launcher the MCP ``run`` tool uses (allowlist from ``vqlab.commands``, the
  box's GPU lease, refusal while an exo instance is placed, paths confined
  to the configured storage roots). Poll ``run_status``; ``stop`` ends it;
* cheap CPU work (header reads, record queries, paired KL over saved arrays,
  static gates) runs in-process and returns its result;
* ``import vqlab.api`` is cheap: no mlx, no tool modules, until a function
  that needs them is called.

``publish`` is deliberately absent, as it is from the MCP server: uploading
is irreversible and is a human's action. ``release_prep`` prints the exact
line for one.

Example::

    from dataclasses import asdict
    from vqlab import api

    hits = api.where_is(name="397B")                       # dict, in-process
    h = api.score(kind="kl-ladder", opts={"cache": "...", "rung": ["a", "b"]})
    api.run_status(run_id=h.run_id)["status"]              # poll; never blocks
    asdict(api.gate(check="bundle", artifact="/models/x")) # GateResult
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Union

from vqlab.ops.errors import LabError

__all__ = [
    "ApiError", "CommandInfo", "RunHandle", "GateResult", "ToolOutput", "CardResult",
    "to_json",
    "commands", "run", "run_status", "stop", "list_runs", "gpu_state",
    "where_is", "list_artifacts", "disk_free", "artifact_info", "artifact_size",
    "fit", "geo_build", "score", "bench",
    "bundle", "gate", "kl_pair", "release_prep", "card",
    "fits", "findings_tail", "next_f_number", "release_f_number", "findings_append",
    "version_stamp",
]

JSONDict = Dict[str, Any]
OptValue = Union[str, int, float, bool, None, Sequence[Union[str, int, float]]]


# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #

# A refusal or failure. ``code`` is stable (NOT_ALLOWED, OUTSIDE_ROOTS,
# EXO_PLACED, GPU_BUSY, NOT_FOUND, INVALID_ARG, FAILED, ...); ``extra`` carries
# JSON-typed detail; ``as_dict()`` is the JSON form. It is the lab library's
# own error type (vqlab.ops.errors.LabError), so a refusal raised anywhere in
# vqlab.ops reaches the caller as-is: one type, no mapping.
ApiError = LabError


@dataclass(frozen=True)
class CommandInfo:
    """One ``vqlab <name>`` command from the registry."""
    name: str
    stage: str
    module: str
    gpu: bool          # may touch the GPU (False: header reads, records, HTTP clients)
    mcp: bool          # launchable through run() / the MCP ``run`` tool
    help: str


@dataclass(frozen=True)
class RunHandle:
    """A detached job launched on this box. Poll ``run_status(run_id=...)``."""
    run_id: str
    run_dir: str
    log_path: str      # the job's combined stdout/stderr (appears once it starts)
    pid: int           # the supervisor's pid (its own process group)
    command: str       # the equivalent shell line, for the record


@dataclass(frozen=True)
class GateResult:
    """A gate's verdict. ``ok`` is the gate's own exit-0; ``status`` is its
    verdict word (PASS / FAIL / STALE / INCONCLUSIVE); ``failures`` are its
    FAIL lines; ``report`` is its full text, exactly as the CLI prints it."""
    check: str
    ok: bool
    status: str
    exit_code: int
    failures: List[str]
    report: str


@dataclass(frozen=True)
class ToolOutput:
    """An in-process tool run whose product is a side effect plus its log."""
    ok: bool
    exit_code: int
    output: str


@dataclass(frozen=True)
class CardResult:
    """A model card's text and what still blocks it from release."""
    text: str
    todos: int
    spellings: List[Dict[str, Any]] = field(default_factory=list)
    releasable: bool = False
    written_to: Optional[str] = None


def to_json(x: Any) -> Any:
    """A return value of this module as plain JSON types (dataclasses become
    dicts; lists and dicts pass through)."""
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return dataclasses.asdict(x)
    if isinstance(x, list):
        return [to_json(v) for v in x]
    return x


# --------------------------------------------------------------------------- #
# Plumbing
# --------------------------------------------------------------------------- #

def _call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run an internal tool; a SystemExit becomes ApiError (a LabError, which
    IS ApiError, passes through)."""
    try:
        return fn(*args, **kwargs)
    except SystemExit as e:
        if e.code in (None, 0):
            raise ApiError("FAILED", f"{getattr(fn, '__name__', fn)} exited early") from None
        raise ApiError("FAILED", str(e.code)) from None


def _need(**named: Any) -> None:
    missing = [k for k, v in named.items() if v is None or v == ""]
    if missing:
        raise ApiError("INVALID_ARG", f"required: {', '.join(missing)}", missing=missing)


def _choice(name: str, value: str, allowed: Sequence[str]) -> None:
    if value not in allowed:
        raise ApiError("INVALID_ARG", f"{name} must be one of {list(allowed)}, got {value!r}",
                       allowed=list(allowed))


def _flags(opts: Optional[Mapping[str, OptValue]]) -> List[str]:
    """{"vq_layers": "0-9", "pack": True, "rung": ["a", "b"]} ->
    ["--vq-layers", "0-9", "--pack", "--rung", "a", "--rung", "b"].
    False / None are omitted; a list repeats the flag."""
    out: List[str] = []
    for k, v in (opts or {}).items():
        if not isinstance(k, str) or not k or k.startswith("-"):
            raise ApiError("INVALID_ARG", f"option names are bare identifiers, got {k!r}")
        flag = "--" + k.replace("_", "-")
        if v is None or v is False:
            continue
        if v is True:
            out.append(flag)
        elif isinstance(v, (list, tuple)):
            for x in v:
                out += [flag, str(x)]
        elif isinstance(v, (str, int, float)):
            out += [flag, str(v)]
        else:
            raise ApiError("INVALID_ARG", f"option {k!r}: unsupported value type "
                                          f"{type(v).__name__}")
    return out


def _capture(main: Callable[[List[str]], Optional[int]], argv: List[str]) -> ToolOutput:
    """Run a tool's main(argv) in-process with its stdout/stderr captured."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            rc = main(argv) or 0
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
            if not isinstance(e.code, int) and e.code is not None:
                buf.write(str(e.code) + "\n")
    return ToolOutput(ok=rc == 0, exit_code=int(rc), output=buf.getvalue())


# --------------------------------------------------------------------------- #
# Commands and runs (the launcher: vqlab.ops.runs)
# --------------------------------------------------------------------------- #

def commands() -> List[CommandInfo]:
    """Every ``vqlab`` command from the registry (vqlab.commands), sorted by
    name. In-process, no GPU."""
    from vqlab.commands import COMMANDS
    return [CommandInfo(name=c.name, stage=c.stage, module=c.module, gpu=c.gpu,
                        mcp=c.mcp, help=c.help)
            for c in sorted(COMMANDS.values(), key=lambda c: c.name)]


def run(*, command: str, args: Sequence[str] = (), tag: str = "",
        retries: Optional[int] = None, force: bool = False) -> RunHandle:
    """Launch ``vqlab <command> <args...>`` detached on this box and return at
    once. The launcher is the MCP ``run`` tool's: only MCP-allowlisted
    commands (never publish), absolute paths must sit under the storage
    roots, refused while an exo instance is placed or the GPU lease is held
    (GPU-free commands skip both). Resumable builds retry from checkpoints.
    Long-running; the job holds the GPU lease unless the command is GPU-free.
    Raises ApiError(NOT_ALLOWED / OUTSIDE_ROOTS / EXO_PLACED / GPU_BUSY ...)."""
    _need(command=command)
    if isinstance(args, str):
        raise ApiError("INVALID_ARG", "args is a sequence of strings, not one string")
    from vqlab.ops import runs
    r = _call(runs.launch, command, [str(a) for a in args], tag=tag,
              retries=retries, force=force)
    return RunHandle(run_id=r["run_id"], run_dir=r["run_dir"],
                     log_path=str(Path(r["run_dir"]) / "job.log"),
                     pid=int(r["supervisor_pid"]), command=r["command"])


def run_status(*, run_id: str, tail: int = 40) -> JSONDict:
    """A run's metadata (status: launching / running / completed / failed /
    stopped / deferred / lost), exit code, attempts and the last ``tail`` log
    lines. In-process, no GPU."""
    _need(run_id=run_id)
    from vqlab.ops import runs
    return _call(runs.status, run_id, tail=tail)


def stop(*, run_id: str) -> JSONDict:
    """SIGTERM a run's process group. Returns {run_id, stopped, ...}."""
    _need(run_id=run_id)
    from vqlab.ops import runs
    return _call(runs.stop, run_id)


def list_runs(*, n: int = 20) -> JSONDict:
    """The ``n`` most recent runs launched on this box (newest first)."""
    from vqlab.ops import runs
    return _call(runs.list_runs, n)


def gpu_state(*, boxes: bool = True) -> JSONDict:
    """Who holds this box's GPU lease, which exo instances are placed, every
    box's reservation (``vqlab reserve``) and, with ``boxes`` and [boxes.*]
    configured, the same for the other boxes over ssh. Check it before
    run(); a held lease or a placement makes run() refuse. No GPU."""
    from vqlab.ops import box
    return _call(box.gpu_state, boxes)


# --------------------------------------------------------------------------- #
# Storage and artifacts (read-only, in-process, no GPU)
# --------------------------------------------------------------------------- #

def where_is(*, name: str, kind: Literal["any", "dir", "file"] = "any",
             max_depth: int = 3) -> JSONDict:
    """Find an artifact, fit, teacher cache or file by (sub)name across the
    configured storage roots. Deterministic: matches, or found=False with the
    roots searched. ``name`` must be >= 3 characters."""
    _choice("kind", kind, ("any", "dir", "file"))
    from vqlab.ops import storage
    return _call(storage.where_is, name, kind=kind, max_depth=max_depth)


def list_artifacts(*, root: str = "", max_depth: int = 3) -> JSONDict:
    """Directories with a config.json under the storage roots (or one root),
    with packed safetensors GiB and whether model.py / README.md exist."""
    from vqlab.ops import storage
    return _call(storage.list_artifacts, root, max_depth=max_depth)


def disk_free() -> JSONDict:
    """Free and total GiB on every configured storage root (a writer that
    runs out mid-run loses hours: check before any multi-hour job)."""
    from vqlab.ops import storage
    return _call(storage.disk_free)


def artifact_info(*, path: str) -> JSONDict:
    """One artifact's registry entry: model_type, VQ geometry mix, text GiB,
    shard fingerprint, bundled runtime (md5/profile/flags) and build record.
    Reads headers and the first MiB of each shard. Does NOT write the
    registry."""
    _need(path=path)
    p = Path(path)
    if not (p / "config.json").is_file():
        raise ApiError("NOT_FOUND", f"no config.json at {p}")
    from vqlab.records import registry
    return _call(registry.entry, p)


def artifact_size(*, path: str) -> JSONDict:
    """An artifact's bytes three ways (text / +tower / +MTP) and the full
    download, from its own headers, plus whether the index's
    metadata.total_size matches the shards (``index``). Same numbers as
    ``vqlab size``."""
    _need(path=path)
    if not Path(path).is_dir():
        raise ApiError("NOT_FOUND", f"no artifact directory at {path}")
    from vqlab.core.artifact import sizes
    from vqlab.core.artifact_size import index_total_size
    s = dict(_call(sizes, path))
    s["mtp_total"] = s["mtp"] + s["mtp_sidecar"]
    s["text_plus_tower"] = s["text"] + s["tower"]
    s["text_plus_tower_plus_mtp"] = s["text"] + s["tower"] + s["mtp_total"]
    try:
        have, want, state = index_total_size(path)
        s["index"] = {"total_size": have, "tensor_bytes": want, "state": state}
    except (OSError, ValueError, KeyError):
        s["index"] = None          # an unindexed (single-shard) artifact
    return s


# --------------------------------------------------------------------------- #
# Long GPU jobs: argv builders over run()
# --------------------------------------------------------------------------- #

def fit(*, kind: Literal["moe", "dense"], src: str, out: str,
        base: Optional[str] = None, vq_layers: Optional[str] = None,
        opts: Optional[Mapping[str, OptValue]] = None, tag: str = "",
        retries: Optional[int] = None) -> RunHandle:
    """Fit VQ codebooks: ``vqlab fit-moe`` (needs base and vq_layers) or
    ``vqlab fit-dense``. ``opts`` are further flags by name ({"k": 256,
    "family": "qwen4_exp"}). GPU, hours; returns a RunHandle at once.
    (fit-ple / fit-additive are not MCP-launchable, so not here.)"""
    _choice("kind", kind, ("moe", "dense"))
    _need(src=src, out=out)
    if kind == "moe":
        _need(base=base, vq_layers=vq_layers)
        argv = ["--base", base, "--src", src, "--out", out, "--vq-layers", vq_layers]
    else:
        if base or vq_layers:
            raise ApiError("INVALID_ARG", "fit-dense takes no base / vq_layers (use opts.layers)")
        argv = ["--src", src, "--out", out]
    return run(command=f"fit-{kind}", args=argv + _flags(opts), tag=tag, retries=retries)


def geo_build(*, artifact: str, teacher: str, family: str, geomap: str, out: str,
              reuse: Optional[str] = None, opts: Optional[Mapping[str, OptValue]] = None,
              tag: str = "", retries: Optional[int] = None) -> RunHandle:
    """Rebuild an artifact under a new per-layer geometry map (``vqlab
    geo-build``): named modules refit from the bf16 teacher, the rest keep
    the shipped bytes. GPU, hours; returns a RunHandle at once."""
    _need(artifact=artifact, teacher=teacher, family=family, geomap=geomap, out=out)
    argv = ["--artifact", artifact, "--teacher", teacher, "--family", family,
            "--geomap", geomap, "--out", out]
    if reuse:
        argv += ["--reuse", reuse]
    return run(command="geo-build", args=argv + _flags(opts), tag=tag, retries=retries)


_SCORE = {"kl": "kl", "kl-ladder": "kl-ladder", "ppl": "score", "tasks": "tasks"}


def score(*, kind: Literal["kl", "kl-ladder", "ppl", "tasks"],
          opts: Optional[Mapping[str, OptValue]] = None,
          args: Sequence[str] = (), tag: str = "") -> RunHandle:
    """Score an artifact: ``kl`` (kl_damage), ``kl-ladder`` (the paired
    3-corpus release gate), ``ppl`` (``vqlab score``) or ``tasks``. Flags
    come from ``opts`` by name, positionals from ``args``. GPU, long;
    returns a RunHandle at once."""
    _choice("kind", kind, tuple(_SCORE))
    return run(command=_SCORE[kind], args=[*map(str, args), *_flags(opts)], tag=tag)


_BENCH = ("decode-timeline", "coverage", "stage-bandwidth", "serve-timeline", "active-bytes")


def bench(*, kind: Literal["decode-timeline", "coverage", "stage-bandwidth",
                           "serve-timeline", "active-bytes"],
          opts: Optional[Mapping[str, OptValue]] = None,
          args: Sequence[str] = (), tag: str = "") -> RunHandle:
    """Launch an MCP-launchable bench (``vqlab <kind>``) detached. GPU
    (active-bytes / stage-bandwidth / serve-timeline are GPU-free and skip
    the lease); returns a RunHandle at once. Speed numbers are only
    meaningful as RATIOS within one session."""
    _choice("kind", kind, _BENCH)
    return run(command=kind, args=[*map(str, args), *_flags(opts)], tag=tag)


# --------------------------------------------------------------------------- #
# Bundle, gates, KL pairing, release (in-process)
# --------------------------------------------------------------------------- #

def bundle(*, artifact: str, dense: bool = False, runtime: Optional[str] = None,
           adopt: str = "") -> ToolOutput:
    """(Re)write the artifact's self-contained model.py from the current
    runtime: ``vqlab bundle`` (MoE) or ``vqlab rebundle-dense``. WRITES into
    the artifact (dense keeps a .pre-resync backup); shipped VQ_* defaults
    are preserved unless named in ``adopt``. In-process, no GPU, seconds.
    Gate it afterwards with gate(check="bundle")."""
    _need(artifact=artifact)
    if not (Path(artifact) / "config.json").is_file():
        raise ApiError("NOT_FOUND", f"no config.json at {artifact}")
    if runtime is not None:
        _choice("runtime", runtime, ("v1.5", "v2"))
    argv = ["--artifact", artifact] + (["--runtime", runtime] if runtime else []) \
        + (["--adopt", adopt] if adopt else [])
    if dense:
        from vqlab.bundle import rebundle_dense as tool
    else:
        from vqlab.bundle import add_model_file as tool
    return _capture(tool.main, argv)


_VERDICTS = ("PASS", "FAIL", "STALE", "INCONCLUSIVE")


def _gate_result(check: str, rc: int, lines: List[str]) -> GateResult:
    status = next((w for ln in lines for w in [ln.split(":")[0].split(" ")[0]]
                   if w in _VERDICTS), "PASS" if rc == 0 else "FAIL")
    return GateResult(check=check, ok=rc == 0, status=status, exit_code=rc,
                      failures=[ln for ln in lines if ln.startswith("FAIL")],
                      report="\n".join(lines) + ("\n" if lines else ""))


def gate(*, check: Literal["bundle", "comparator"], artifact: str,
         teacher: Optional[str] = None, strict: bool = False) -> GateResult:
    """Run a static gate in-process (headers and source text only, no GPU,
    seconds):

    * ``bundle``     -- ``vqlab check-bundle``: the bundled model.py carries
      the current runtime (STALE passes unless ``strict``);
    * ``comparator`` -- ``vqlab check-comparator``: a comparator holds the
      same core tensor set as ``teacher``.

    The model-loading gates (check-release, check, smoke, verify) need the
    model resident: launch them with run(command="check-release", ...)."""
    _choice("check", check, ("bundle", "comparator"))
    _need(artifact=artifact)
    if not Path(artifact).is_dir():
        raise ApiError("NOT_FOUND", f"no artifact directory at {artifact}")
    if check == "bundle":
        from vqlab.gate import check_bundle
        rc, lines = _call(check_bundle.check, artifact, strict)
    else:
        _need(teacher=teacher)
        if not Path(teacher).is_dir():
            raise ApiError("NOT_FOUND", f"no teacher directory at {teacher}")
        from vqlab.gate import check_comparator
        rc, lines = _call(check_comparator.check, artifact, teacher)
    return _gate_result(check, rc, lines)


def kl_pair(*, per_pos_dir: str, arm: str, ref: str, corpora: Sequence[str],
            allow_unverified: bool = False) -> JSONDict:
    """Paired KL of rung ``arm`` minus rung ``ref`` (negative = arm better)
    from their saved per-position arrays (``<rung>__<corpus>.safetensors``
    in ``per_pos_dir``), the same test as ``vqlab kl-pair``. Refuses a
    pairing whose sidecars do not prove the same cache and numerics build
    unless ``allow_unverified``. CPU, in-process, seconds; zero GPU cost."""
    _need(per_pos_dir=per_pos_dir, arm=arm, ref=ref)
    if isinstance(corpora, str) or not corpora:
        raise ApiError("INVALID_ARG", "corpora is a non-empty list of corpus names")
    from vqlab.score import kl_pair as tool
    rows = [_call(tool.pair_one, per_pos_dir, arm, ref, c, allow_unverified)
            for c in corpora]
    return {"arm": arm, "ref": ref, "per_pos_dir": per_pos_dir, "corpora": rows,
            "verified": not any(r["problems"] for r in rows),
            "worst_abs_pct": max(abs(r["pct"]) for r in rows)}


def release_prep(*, artifact: str, repo: str, hash_weights: bool = False) -> JSONDict:
    """Everything before an upload that needs no model: sizes three ways,
    local junk files, a diff against the Hub repo (network; small files by
    git blob hash, weights by size or sha256 with ``hash_weights``), and the
    exact ``vqlab publish`` line for a human to run. Uploads nothing. The
    release gate itself loads the model: run(command="check-release", ...)."""
    _need(artifact=artifact, repo=repo)
    art = Path(artifact)
    if not (art / "config.json").is_file():
        raise ApiError("NOT_FOUND", f"no config.json at {art}")
    from vqlab.ship import release_prep as tool
    sz = artifact_size(path=artifact)
    junk = [str(Path(j).relative_to(art)) for j in tool._junk(art)]
    changed, only_hub, exists = _call(tool.hub_diff, art, repo, hash_weights)
    ready_line = None if (exists and not changed) else \
        tool.publish_line(art, repo, changed, exists)
    return {"artifact": str(art), "repo": repo, "sizes": sz, "junk": junk,
            "hub_exists": exists, "changed": list(changed), "only_on_hub": list(only_hub),
            "weights_compared_by": "sha256" if hash_weights else "size",
            "publish_line": ready_line,
            "note": "publish is a human action; run the release gate before it"}


def card(*, artifact: str, kl: Sequence[str], this: str, out: Optional[str] = None,
         opts: Optional[Mapping[str, OptValue]] = None) -> CardResult:
    """Build a model card from the artifact's config, the kl-ladder JSON(s)
    and its build record (``vqlab card``). ``this`` names the KL rung that IS
    the artifact. Returns the text; writes it only to ``out`` when given.
    ``opts``: base_model, runtime, label, extra, extra_md, license.
    In-process, no GPU. releasable=False while TODO blocks or British
    spellings remain."""
    _need(artifact=artifact, this=this)
    if isinstance(kl, str) or not kl:
        raise ApiError("INVALID_ARG", "kl is a non-empty list of kl-ladder JSON paths")
    if opts and ("out" in opts or "draft" in opts):
        raise ApiError("INVALID_ARG", "pass out= directly; draft does not apply here")
    from vqlab.gate import spelling
    from vqlab.ship import card as tool
    argv = ["--artifact", artifact, "--this", this]
    for k in kl:
        argv += ["--kl", str(k)]
    try:
        ns = tool.parser().parse_args(argv + _flags(opts))
    except SystemExit:
        raise ApiError("INVALID_ARG", "card options did not parse",
                       allowed=["base_model", "runtime", "label", "extra",
                                "extra_md", "license"]) from None
    text = _call(tool.build, ns)
    text_spell = spelling.scan(text)
    spell = [{"line": text.count("\n", 0, off) + 1, "word": w, "us": us}
             for off, w, us in text_spell]
    todos = tool.todo_count(text)
    if out:
        Path(out).write_text(text)
    return CardResult(text=text, todos=todos, spellings=spell,
                      releasable=not todos and not spell, written_to=out)


# --------------------------------------------------------------------------- #
# Records: fit store, findings log, numerics stamp
# --------------------------------------------------------------------------- #

def fits(*, family: Optional[str] = None, teacher: Optional[str] = None,
         layers: Optional[Sequence[int]] = None, proj: Optional[str] = None,
         d: Optional[int] = None, K: Optional[int] = None,
         recipe_known: Optional[bool] = None, root: Optional[str] = None) -> List[JSONDict]:
    """Fitted modules in the fit store's index (``vqlab fits list``),
    filtered; ``teacher`` matches as a substring. ``root`` defaults to the
    first configured fit-store root. Reads the index only; no GPU."""
    from vqlab.core import fitstore as fs
    if root:
        r = Path(root)
    else:
        roots = fs.roots()
        if not roots:
            raise ApiError("NOT_FOUND", "no fit store configured (vqlab.config fit_store)")
        r = Path(roots[0])
    recs = fs.read_index(r)
    return fs.query(recs, family=family, teacher=teacher,
                    layers=set(layers) if layers is not None else None,
                    proj=proj, d=d, K=K, recipe_known=recipe_known)


def findings_tail(*, n: int = 10) -> JSONDict:
    """The last ``n`` F-entry headlines of the findings log, with line
    numbers. Read-only."""
    from vqlab.ops import findings
    return _call(findings.findings_tail, n)


def next_f_number(*, reserve: bool = False, owner: str = "") -> JSONDict:
    """The next free F-number. ``reserve=True`` CLAIMS it atomically (a lock
    file, 6 h TTL) -- do that before a long experiment; a bare read races
    any concurrent session."""
    from vqlab.ops import findings
    return _call(findings.next_f_number, reserve=reserve, owner=owner)


def release_f_number(*, n: int) -> JSONDict:
    """Give back an F-number claimed with next_f_number(reserve=True) whose
    experiment was abandoned. Returns {released, was_held}."""
    from vqlab.ops import findings
    return _call(findings.release_f_number, n)


def findings_append(*, headline: str, artifact: str, instrument: str, prediction: str,
                    measured: str, verdict: str, body: str = "", corrects: str = "",
                    use_reserved: int = 0) -> JSONDict:
    """Append the next F-entry to the findings log -- the only library writer
    of the measured record. Every field is required; ``prediction`` (the
    pre-registered one) is written verbatim; ``verdict`` is one of CONFIRMED /
    FALSIFIED / VOID / NULL / CORRECTS (CORRECTS needs ``corrects="F<n>"``).
    ``use_reserved`` consumes a number this caller reserved (and releases
    its lock); omitted, a fresh number is allocated. WRITES the log."""
    from vqlab.ops import findings
    return _call(findings.findings_append, headline, artifact, instrument, prediction,
                 measured, verdict, body=body, corrects=corrects, use_reserved=use_reserved)


def version_stamp(*, model_type: Optional[str] = None, family: Optional[str] = None,
                  model_dir: Optional[str] = None) -> JSONDict:
    """This interpreter's numerics build for a model_type (or the one
    ``model_dir``'s config names): mlx / mlx-lm / knurlogic versions, the
    resolved arch file and its sha256, the scorer variant. What every score
    is stamped with; two scores pair only if their stamps agree. May import
    the arch's package to locate it; no GPU. The CLI serves Knurlogic's
    architectures; call ``vqlab.use_knurlogic_arch()`` first to stamp the
    same build a ``vqlab`` command would."""
    from vqlab.family import version_stamp as tool
    return _call(tool.stamp, model_type, family=family, model_dir=model_dir)

