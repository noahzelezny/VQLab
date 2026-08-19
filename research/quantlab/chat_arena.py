#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Side-by-side BLIND chat arena for two local MLX models.

WHY. Every automated instrument returned parity between gemma-small
(2.25bpw VQ of 26b-a4b) and e4b-8bit, and Noah's read is the one signal we
have not captured — but a terminal is a bad place to form it (no wrapping,
token caps, and you can see which model you are talking to, so the
impression is not blind).

This serves both models to a browser, side by side, with the SAME prompt
going to both. Assignment to pane A/B is re-randomized every turn, so
position bias cannot accumulate. Identity stays hidden until you vote; the
vote plus the true identity is appended to winrate/human_verdicts.jsonl,
which makes your impressions a paired instrument we can actually count
(sign test on non-ties) instead of an anecdote.

    ./chat_arena.py            # then open http://localhost:8770
"""
import http.server
import json
import pathlib
import random
import socketserver
import threading

HERE = pathlib.Path(__file__).parent
E = pathlib.Path("/Volumes/Thunderbay SSD/Exo Models")
MODELS = {
    "gemma-small (VQ 2.25bpw of 26b-a4b, 9.4G)": E / "gemma26b-rungs" / "vq-K256-d4",
    "e4b-8bit (community incumbent, 8.4G)": E / "mlx-community--gemma-4-e4b-it-8bit",
}
NAMES = list(MODELS)
VERDICTS = HERE / "winrate" / "human_verdicts.jsonl"

_loaded, _lock = {}, threading.Lock()


def get(name):
    with _lock:
        if name not in _loaded:
            from mlx_lm.utils import load, load_model, load_tokenizer
            p = MODELS[name]
            try:
                m, t, _ = load(str(p), return_config=True)
            except ValueError:
                m, _ = load_model(p, strict=False)
                t = load_tokenizer(p)
            _loaded[name] = (m, t)
        return _loaded[name]


def gen_stream(name, messages, max_tokens):
    import mlx.core as mx
    from mlx_lm import stream_generate
    model, tok = get(name)
    try:
        text = tok.apply_chat_template(messages, add_generation_prompt=True,
                                       tokenize=False, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(messages, add_generation_prompt=True,
                                       tokenize=False)
    stop = ("<turn|>", "<end_of_turn>", "<|im_end|>")
    buf = ""
    for r in stream_generate(model, tok, prompt=text, max_tokens=max_tokens):
        buf += r.text
        if any(s in buf for s in stop):
            for s in stop:
                buf = buf.split(s)[0]
            yield r.text.split("<")[0]
            break
        yield r.text
    mx.clear_cache()


PAGE = r"""<!doctype html><meta charset=utf-8><title>VQ chat arena</title>
<style>
:root{--bg:#faf9f7;--fg:#1a1a19;--mut:#6b6a65;--line:#e1e0d9;--card:#fff;--acc:#1d9e75}
@media(prefers-color-scheme:dark){:root{--bg:#141413;--fg:#f0efec;--mut:#a1a09a;--line:#2c2c2a;--card:#1e1e1c}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 -apple-system,BlinkMacSystemFont,system-ui,sans-serif}
header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;gap:14px;align-items:center;flex-wrap:wrap}
h1{font-size:16px;font-weight:500;margin:0}
.sp{flex:1}
button{font:inherit;font-size:14px;padding:7px 14px;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:8px;cursor:pointer}
button:hover{border-color:var(--mut)}
button.p{background:var(--acc);color:#fff;border-color:var(--acc)}
main{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:16px;align-items:start}
.pane{background:var(--card);border:1px solid var(--line);border-radius:12px;min-height:220px;display:flex;flex-direction:column}
.ph{padding:10px 16px;border-bottom:1px solid var(--line);font-size:13px;color:var(--mut);display:flex;gap:8px;align-items:center}
.id{font-weight:500;color:var(--fg)}
.body{padding:16px;white-space:pre-wrap;overflow-wrap:anywhere;flex:1}
.msg{margin-bottom:18px}
.u{color:var(--mut);font-size:13px;border-left:2px solid var(--line);padding-left:10px;margin-bottom:10px}
footer{position:sticky;bottom:0;background:var(--bg);border-top:1px solid var(--line);padding:12px 16px;display:flex;gap:10px}
textarea{flex:1;font:inherit;padding:10px 12px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);resize:vertical;min-height:52px}
.vote{padding:10px 16px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:14px;color:var(--mut)}
.tag{font-size:12px;padding:2px 8px;border-radius:20px;background:var(--line);color:var(--fg)}
@media(max-width:860px){main{grid-template-columns:1fr}}
</style>
<header><h1>VQ chat arena</h1><span class=tag id=st>blind</span><span class=sp></span>
<button id=reveal>reveal</button><button id=reset>new conversation</button></header>
<main>
<div class=pane><div class=ph>pane <span class=id>A</span> <span id=na></span></div><div class=body id=a></div></div>
<div class=pane><div class=ph>pane <span class=id>B</span> <span id=nb></span></div><div class=body id=b></div></div>
</main>
<div class=vote>this exchange: <button data-v=A>A better</button><button data-v=B>B better</button>
<button data-v=tie>tie</button><span id=vs></span></div>
<footer><textarea id=q placeholder="Ask both models the same thing…  (Enter to send, Shift+Enter for newline)"></textarea><button class=p id=send>send</button></footer>
<script>
let hist={A:[],B:[]}, map=null, revealed=false, busy=false;
const $=s=>document.querySelector(s);
function paint(){ $('#na').textContent = revealed&&map ? map.A : ''; $('#nb').textContent = revealed&&map ? map.B : '';
  $('#st').textContent = revealed?'revealed':'blind'; }
$('#reveal').onclick=()=>{revealed=!revealed;paint()};
$('#reset').onclick=()=>{hist={A:[],B:[]};map=null;revealed=false;$('#a').innerHTML='';$('#b').innerHTML='';paint()};
$('#q').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}});
$('#send').onclick=send;
document.querySelectorAll('.vote button').forEach(b=>b.onclick=async()=>{
  if(!map){return}
  await fetch('/vote',{method:'POST',body:JSON.stringify({vote:b.dataset.v,map:map,turns:hist.A.length})});
  $('#vs').textContent='recorded ✓'; setTimeout(()=>$('#vs').textContent='',1800);
});
async function send(){
  if(busy)return; const q=$('#q').value.trim(); if(!q)return; busy=true; $('#q').value='';
  for(const p of ['A','B']){ hist[p].push({role:'user',content:q});
    $('#'+p.toLowerCase()).innerHTML+='<div class="msg"><div class=u>'+esc(q)+'</div><span id=cur'+p+'></span></div>'; }
  const r=await fetch('/chat',{method:'POST',body:JSON.stringify({hist:hist,map:map})});
  const rd=r.body.getReader(), dec=new TextDecoder(); let buf='';
  const got={A:'',B:''};
  while(true){const {done,value}=await rd.read(); if(done)break; buf+=dec.decode(value,{stream:true});
    let i; while((i=buf.indexOf('\n'))>=0){ const line=buf.slice(0,i); buf=buf.slice(i+1); if(!line)continue;
      const m=JSON.parse(line);
      if(m.map){map=m.map;paint();continue}
      got[m.pane]+=m.t; document.getElementById('cur'+m.pane).textContent=got[m.pane]; } }
  hist.A.push({role:'assistant',content:got.A}); hist.B.push({role:'assistant',content:got.B});
  busy=false; $('#q').focus();
}
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
paint();
</script>"""


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(PAGE.encode())

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/vote":
            VERDICTS.parent.mkdir(exist_ok=True)
            with open(VERDICTS, "a") as f:
                f.write(json.dumps(req) + "\n")
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
            return
        # /chat — re-randomize the A/B assignment EVERY turn
        m = req.get("map")
        if not m:
            pair = NAMES[:] 
            random.shuffle(pair)
            m = {"A": pair[0], "B": pair[1]}
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        self.wfile.write((json.dumps({"map": m}) + "\n").encode())
        self.wfile.flush()
        for pane in ("A", "B"):
            for tok in gen_stream(m[pane], req["hist"][pane], 900):
                self.wfile.write((json.dumps({"pane": pane, "t": tok}) + "\n").encode())
                self.wfile.flush()


class S(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


print("chat arena on http://localhost:8770  (blind by default; votes -> "
      f"{VERDICTS})")
S(("127.0.0.1", 8770), H).serve_forever()
