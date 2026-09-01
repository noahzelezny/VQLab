"""Referee: score the same corpus slice on whatever model exo has loaded."""
import json, sys, urllib.request, pathlib
model = sys.argv[1]
corpus = pathlib.Path(__file__).parent / "referee_corpus.txt"
payload = {
    "model": model,
    "messages": [{"role": "user", "content": corpus.read_text()}],
    "echo_score": True, "max_tokens": 1, "temperature": 0,
}
req = urllib.request.Request("http://localhost:52415/v1/chat/completions",
    data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=3600))
content = r["choices"][0]["message"].get("content") or ""
try:
    print(json.dumps({"model": model, **json.loads(content)}, indent=1))
except Exception:
    print("RAW:", content[:400])
