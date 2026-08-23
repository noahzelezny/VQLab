#!/bin/sh
# Gate: refuse to run a chain on scripts that drift from repo HEAD.
# Stale scripts on a second box are silent divergence — run this in every
# chain preamble.
# Usage: ./check_scripts_sync.sh [file ...]   (defaults to the fit/pack/gate set)
set -e
cd "$(dirname "$0")/../src/moemash"
if command -v md5 >/dev/null 2>&1; then MD5="md5 -q"; else MD5="md5sum"; fi
sum() { $MD5 "$@" 2>/dev/null | awk '{print $1}'; }
FILES="${*:-vq_397b_codes.py fit_dense_vq.py vq_switch.py vq_dense.py vq_pack.py add_model_file.py verify_artifact.py pack_artifact.py pack_dense.py check_release.py check_bundle.py}"
fail=0
for f in $FILES; do
  wt=$(sum "$f"); [ -n "$wt" ] || wt=MISSING
  hd=$(git show HEAD:"src/moemash/$f" 2>/dev/null | $MD5 | awk '{print $1}')
  if [ "$wt" != "$hd" ]; then echo "SYNC FAIL: $f (wt=$wt head=$hd)"; fail=1; fi
done
[ $fail -eq 0 ] && echo "SYNC OK: all scripts match repo HEAD" || exit 1
