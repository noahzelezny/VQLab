#!/usr/bin/env python3
"""`~` as "approximately" is a strikethrough delimiter on the Hub.

Two `~` in the same block render everything between them struck through, which
is what happened to the runtime-refresh bullet. Fixes:

  "up to ~1.46x"  -> "up to 1.46x"   (the tilde was redundant after "up to")
  "~2%", "~85%"   -> "≈2%",  "≈85%"  (unambiguous, not a delimiter)

Tildes inside code spans/fences are left alone — they don't render as markdown.
"""
import re, pathlib, sys

D = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")

def fix(text):
    out, n = [], 0
    fence = False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            fence = not fence
            out.append(line); continue
        if fence:
            out.append(line); continue
        # protect code spans
        spans = []
        def stash(m):
            spans.append(m.group(0)); return f"\x00{len(spans)-1}\x00"
        s = re.sub(r"`[^`]*`", stash, line)
        s, a = re.subn(r"\b(up to )~(?=\d)", r"\1", s)
        s, b = re.subn(r"~(?=\d)", "≈", s)
        n += a + b
        s = re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], s)
        out.append(s)
    return "\n".join(out), n

total = 0
for p in sorted(D.glob("*.md")):
    t = p.read_text()
    new, n = fix(t)
    new = new.replace("%%", "%")           # stray double percent
    if new != t:
        p.write_text(new); total += n
        print(f"  {p.stem:34} {n} tilde(s)")
print(f"\n{total} fixed")
