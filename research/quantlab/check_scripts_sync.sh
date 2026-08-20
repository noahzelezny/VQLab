#!/bin/sh
# Gate: refuse to run a chain on scripts that drift from repo HEAD.
# Usage: ./check_scripts_sync.sh [file ...]   (defaults to the fit/pack/gate set)
set -e
cd "$(dirname "$0")"
FILES="${*:-vq_397b_codes.py vq_switch.py vq_pack.py add_model_file.py verify_artifact.py pack_artifact.py pack_dense.py check_vision.py check_release.py}"
fail=0
for f in $FILES; do
  wt=$(md5 -q "$f" 2>/dev/null || echo MISSING)
  hd=$(git show HEAD:"$f" 2>/dev/null | md5 -q)
  if [ "$wt" != "$hd" ]; then echo "SYNC FAIL: $f (wt=$wt head=$hd)"; fail=1; fi
done
[ $fail -eq 0 ] && echo "SYNC OK: all scripts match repo HEAD" || exit 1
