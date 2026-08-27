#!/usr/bin/env python3
"""Render DRAFT.md as a self-contained HTML page for publishing.

Figures are inlined as data URIs -- the artifact CSP blocks every external
host, so a linked PNG would silently fail to load.
"""
import base64, re, os, markdown

SRC = "DRAFT.md"
OUT = "paper.html"

md = open(SRC, encoding="utf-8").read()
title_line = md.split("\n")[0].lstrip("# ").strip()
body_md = "\n".join(md.split("\n")[1:])

def embed(m):
    alt, path = m.group(1), m.group(2)
    if not os.path.exists(path):
        return m.group(0)
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return f'<figure><img src="data:image/png;base64,{b64}" alt="{alt}"><figcaption>{alt}</figcaption></figure>'

body_md = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', embed, body_md)

html = markdown.markdown(body_md, extensions=["tables", "fenced_code", "attr_list"])
html = html.replace("<table>", '<div class="tw"><table>').replace("</table>", "</table></div>")

# section index from the h2s
toc = "".join(
    f'<li><a href="#{re.sub(r"[^a-z0-9]+","-",t.lower()).strip("-")}">{t}</a></li>'
    for t in re.findall(r"<h2>(.*?)</h2>", html))
def anchor(m):
    t = m.group(1); i = re.sub(r"[^a-z0-9]+", "-", re.sub("<.*?>", "", t).lower()).strip("-")
    return f'<h2 id="{i}">{t}</h2>'
html = re.sub(r"<h2>(.*?)</h2>", anchor, html)

CSS = """
:root{
  --ground:#FAFBFC; --panel:#FFFFFF; --ink:#151A20; --muted:#5C6672;
  --rule:#E0E5EA; --accent:#0F7C87; --accent-soft:#E9F3F4; --shadow:rgba(19,28,38,.06);
}
@media (prefers-color-scheme:dark){ :root:not([data-theme="light"]){
  --ground:#0D1116; --panel:#12181F; --ink:#E4E9EE; --muted:#8D97A3;
  --rule:#222A33; --accent:#3FB8C4; --accent-soft:#10262A; --shadow:rgba(0,0,0,.4);
}}
:root[data-theme="dark"]{
  --ground:#0D1116; --panel:#12181F; --ink:#E4E9EE; --muted:#8D97A3;
  --rule:#222A33; --accent:#3FB8C4; --accent-soft:#10262A; --shadow:rgba(0,0,0,.4);
}
*{box-sizing:border-box}
body{
  background:var(--ground); color:var(--ink); margin:0;
  font-family:ui-serif,"Iowan Old Style",Charter,"Source Serif 4",Georgia,serif;
  font-size:18px; line-height:1.65; -webkit-font-smoothing:antialiased;
}
.wrap{display:grid; grid-template-columns:1fr; gap:0; max-width:1180px; margin:0 auto; padding:0 24px}
@media(min-width:1080px){ .wrap{grid-template-columns:210px minmax(0,1fr); gap:56px} }
nav.toc{display:none}
@media(min-width:1080px){
  nav.toc{display:block; position:sticky; top:0; align-self:start; padding:88px 0 40px;
    max-height:100vh; overflow-y:auto}
  nav.toc ol{list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:9px}
  nav.toc a{font-family:ui-monospace,"SF Mono",Menlo,monospace; font-size:11.5px;
    letter-spacing:.03em; color:var(--muted); text-decoration:none; display:block; line-height:1.4}
  nav.toc a:hover,nav.toc a:focus-visible{color:var(--accent)}
}
main{max-width:69ch; padding:88px 0 120px; min-width:0}
.eyebrow{font-family:ui-monospace,"SF Mono",Menlo,monospace; font-size:11.5px;
  letter-spacing:.13em; text-transform:uppercase; color:var(--accent); margin:0 0 18px}
h1{font-size:2.35rem; line-height:1.16; margin:0 0 14px; font-weight:600;
  letter-spacing:-.015em; text-wrap:balance}
.byline{font-family:ui-monospace,"SF Mono",Menlo,monospace; font-size:12.5px;
  color:var(--muted); margin:0 0 52px; letter-spacing:.02em}
h2{font-size:1.4rem; font-weight:600; letter-spacing:-.01em; margin:64px 0 4px;
  padding-top:22px; border-top:2px solid var(--accent); text-wrap:balance; scroll-margin-top:24px}
h3{font-size:1.08rem; font-weight:600; margin:38px 0 2px; color:var(--ink); text-wrap:balance}
h2+p,h3+p{margin-top:12px}
p{margin:0 0 18px}
strong{font-weight:600}
a{color:var(--accent)}
code{font-family:ui-monospace,"SF Mono",Menlo,monospace; font-size:.86em;
  background:var(--accent-soft); padding:.12em .34em; border-radius:3px}
pre{background:var(--panel); border:1px solid var(--rule); border-radius:6px;
  padding:16px 18px; overflow-x:auto; font-size:13px; line-height:1.55}
pre code{background:none; padding:0; font-size:inherit}
blockquote{margin:24px 0; padding:2px 0 2px 20px; border-left:3px solid var(--rule);
  color:var(--muted)}
blockquote p:last-child{margin-bottom:0}
ul,ol{margin:0 0 18px; padding-left:22px}
li{margin-bottom:7px}
.tw{overflow-x:auto; margin:26px 0; border:1px solid var(--rule); border-radius:7px;
  background:var(--panel); box-shadow:0 1px 2px var(--shadow)}
table{border-collapse:collapse; width:100%; font-family:ui-monospace,"SF Mono",Menlo,monospace;
  font-size:12.5px; font-variant-numeric:tabular-nums}
th{text-align:left; font-weight:600; color:var(--muted); text-transform:uppercase;
  letter-spacing:.06em; font-size:10.5px; padding:11px 14px; border-bottom:1px solid var(--rule);
  white-space:nowrap}
td{padding:9px 14px; border-bottom:1px solid var(--rule); vertical-align:top}
tr:last-child td{border-bottom:none}
td strong{color:var(--accent); font-weight:600}
figure{margin:34px 0}
figure img{max-width:100%; height:auto; display:block; border:1px solid var(--rule);
  border-radius:7px; background:#fff}
figcaption{font-family:ui-monospace,"SF Mono",Menlo,monospace; font-size:11.5px;
  color:var(--muted); margin-top:10px; letter-spacing:.02em}
hr{border:none; border-top:1px solid var(--rule); margin:44px 0}
:focus-visible{outline:2px solid var(--accent); outline-offset:3px; border-radius:2px}
@media (prefers-reduced-motion:no-preference){ nav.toc a{transition:color .15s ease} }
@media(max-width:640px){ body{font-size:17px} h1{font-size:1.85rem} main{padding:56px 0 80px} }
"""

CANONICAL_URL = "https://thedrainflorist.com/ai/papers/data-free-vector-quantization/"
PAGE_DESCRIPTION = "Data-free vector quantization beats affine quantization at matched bytes below 6 bits, measured on Apple Silicon."
OG_IMAGE = "https://thedrainflorist.com/images/below-six-bits-og.png"

page = f"""<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title_line}</title>
<link rel="canonical" href="{CANONICAL_URL}">
<meta name="description" content="{PAGE_DESCRIPTION}">
<meta property="og:title" content="{title_line}">
<meta property="og:description" content="{PAGE_DESCRIPTION}">
<meta property="og:type" content="article">
<meta property="og:url" content="{CANONICAL_URL}">
<meta property="og:image" content="{OG_IMAGE}">
<meta property="og:image:width" content="1680">
<meta property="og:image:height" content="1040">
<meta name="twitter:card" content="summary_large_image">
<style>{CSS}</style>
<div class="wrap">
<nav class="toc" aria-label="Sections"><ol>{toc}</ol></nav>
<main>
<p class="eyebrow">Quantization &middot; Apple Silicon &middot; 2026</p>
<h1>{title_line}</h1>
{html}
</main>
</div>
"""
open(OUT, "w", encoding="utf-8").write(page)
print(f"wrote {OUT} — {len(page)/1024:.0f} KB, {len(toc.split('<li>'))-1} sections")
