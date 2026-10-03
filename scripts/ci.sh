#!/usr/bin/env bash
# The whole CI pipeline, runnable locally: `scripts/ci.sh`.
# .github/workflows/ci.yml runs exactly this script, so a green local run is a
# green CI run. Needs Apple Silicon (MLX) and Python >= 3.12.
#
#   1. lint         ruff (pyflakes + syntax) over code, tests, scripts
#   2. install      build a wheel and install it into a fresh venv
#   3. package      the INSTALLED package: CLI entry point, shipped data files
#   4. tests        pytest (the default set: pyproject deselects @pytest.mark.lab, which
#                   reads real lab artifacts; storage here is a throwaway dir, so they
#                   could only skip. On a lab box run them by hand: `pytest -m lab`)
#   5. selftest     every gate, both directions (`vqlab selftest`), from the installed wheel
#
# Runs isolated from this machine's lab setup: no vqlab config file, storage
# under a throwaway directory. PYTHON picks the interpreter (default python3.12).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3.12}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/vqlab-ci.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

# Isolation: forget any VQLAB_* settings and the user's config file.
while IFS= read -r v; do unset "$v"; done < <(env | sed -n 's/^\(VQLAB_[A-Z_]*\)=.*/\1/p')
export VQLAB_CONFIG="$WORK/no-config.toml"
export VQLAB_SCRATCH="$WORK/scratch" VQLAB_FIT_STORE="$WORK/fits"
export VQLAB_MODELS_DIR="$WORK/models" VQLAB_TEACHERS_DIR="$WORK/teachers"
export VQLAB_GPU_LEASE="$WORK/gpu.lease" VQLAB_QUEUE_DIR="$WORK/queues"
mkdir -p "$VQLAB_SCRATCH" "$VQLAB_FIT_STORE"

step() { printf '\n==> %s\n' "$*"; }

step "environment"
"$PY" --version
uname -sm

step "install (wheel into a fresh venv)"
"$PY" -m venv "$WORK/venv"
VPY="$WORK/venv/bin/python"
"$VPY" -m pip install --quiet --upgrade pip build
"$VPY" -m build --wheel --outdir "$WORK/dist" "$ROOT" >/dev/null
rm -rf "$ROOT/build" "$ROOT"/src/*.egg-info
WHEEL="$(ls "$WORK"/dist/vqlab-*.whl)"
"$VPY" -m pip install --quiet "$WHEEL[test]"
"$VPY" - <<'EOF'
import mlx.core as mx
info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
print(f"mlx {mx.__version__} | {info.get('device_name')} ({info.get('architecture')}) "
      f"| {info.get('memory_size', 0) / 2**30:.0f} GiB | metal {mx.metal.is_available()}")
EOF

step "lint"
"$WORK/venv/bin/ruff" check "$ROOT/src" "$ROOT/tests" "$ROOT/scripts" "$ROOT/research/paper"

step "package (the installed wheel, run from outside the checkout)"
(
  cd "$WORK"
  "$WORK/venv/bin/vqlab" --help >/dev/null
  "$VPY" - <<'EOF'
import importlib.resources as r
import json

pkg = r.files("vqlab")
need = ["score/referee/referee_corpus.txt", "runtime/equivalent_revisions.json",
        "agents/gui_static/index.html", "agents/gui_static/bench/index.html"]
reg = json.loads((pkg / "runtime/equivalent_revisions.json").read_text())
need += ["runtime/" + rev["file"] for rev in reg["revisions"]]
missing = [n for n in need if not (pkg / n).is_file()]
if missing:
    raise SystemExit(f"wheel is missing package data: {missing}")
print(f"wheel carries all {len(need)} required data files")
EOF
)

step "tests"
cd "$ROOT"
"$VPY" -m pytest -q -p no:cacheprovider

step "selftest (the INSTALLED wheel, run from outside the checkout: what pip users get)"
(cd "$WORK" && env -u PYTHONPATH "$WORK/venv/bin/vqlab" selftest)

printf '\nCI passed.\n'
