#!/bin/sh
# Gate: a model card must not ship with an unfilled placeholder token.
# Exists because MODEL_CARD_397B_G.md carries __PREDECESSOR_REVISION__, which
# is only knowable at upload time -- exactly the kind of token that survives
# into a published card because everyone assumed someone else filled it.
# Usage: ./check_card_placeholders.sh MODEL_CARD_*.md
[ $# -eq 0 ] && { echo "usage: $0 <card.md>..."; exit 2; }
fail=0
for f in "$@"; do
  hits=$(grep -n "__[A-Z][A-Z0-9_]*__\|<TODO\|TKTK\|XXX" "$f" 2>/dev/null)
  if [ -n "$hits" ]; then
    echo "PLACEHOLDER in $f:"; echo "$hits" | sed 's/^/    /'; fail=1
  fi
done
[ $fail -eq 0 ] && echo "CARD PLACEHOLDER CHECK OK: $# file(s) clean" || exit 1
