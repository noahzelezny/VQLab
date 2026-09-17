"""Rule-5 reference: a DIRECT full-model resident forward on a qwen3_5_moe
artifact, chunked at --chunk with ONE shared cache list -- the house standard
the streamed scorer must reproduce to all printed decimals. ppl math is
copied verbatim from stream_score.main."""
import argparse, math, sys, json, pathlib
sys.path.insert(0, "/Users/noahzelezny/Documents/AgenicAI/vqlab/src/vqlab")
import mlx.core as mx
import runtime_load
from mlx_lm.utils import load_tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--corpus", required=True)
ap.add_argument("--tokens", type=int, default=2048)
ap.add_argument("--chunk", type=int, default=512)
a = ap.parse_args()

mp = pathlib.Path(a.model)
model, cfg = runtime_load.load_for_family("qwen3_5", mp, lazy=False)
print(runtime_load.resolved_runtime_note(model), flush=True)
tok = load_tokenizer(mp)
ids = tok.encode(open(a.corpus).read())[: a.tokens + 1]
bos = getattr(tok, "bos_token_id", None)
if bos is not None and (not ids or ids[0] != bos):
    ids = [bos] + ids[: a.tokens]

x = mx.array([ids[:-1]])
S = x.shape[1]
C = a.chunk
lm = getattr(model, "language_model", model)
caches = lm.make_cache()
lg = []
for s0 in range(0, S, C):
    out = model(x[:, s0:min(s0 + C, S)], cache=caches)
    lg.append(out.astype(mx.float32)[0])
    mx.eval(lg[-1])
logits = mx.concatenate(lg, axis=0) if len(lg) > 1 else lg[0]

tgt = mx.array(ids[1:])
lse = mx.logsumexp(logits, axis=-1)
pk = mx.take_along_axis(logits, tgt[:, None].astype(mx.int64), axis=-1)[:, 0]
nll = lse - pk
ppl = math.exp(float(mx.mean(nll).item()))
print(json.dumps({"model": str(mp), "corpus": a.corpus, "tokens": len(ids) - 1,
                  "chunk": C, "ppl": round(ppl, 6), "direct": True}))
