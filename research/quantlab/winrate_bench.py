#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Blind paired win-rate on long-form literary generation.

WHY THIS EXISTS. gemma-4's two valid instruments disagree: litbench
(generative MC) says VQ-K2048 is at the bf16 ceiling (86.54 vs 84.62, n=104,
saturated), while KL says it diverges 4.2x more than 8-bit from bf16
(1856 vs 441 mnats), a gap the domain scan (CRUSH_RESULTS 08-18) proved is
UNIFORM across general/literary/code — so it is not hiding in any domain,
but nothing tells us whether it MATTERS. Single-token instruments cannot
answer that: routing flips that swap one plausible token for another are
invisible to MC accuracy and heavily punished by KL, symmetrically.

This measures the thing the sidecar actually ships: PROSE. Same prompts to
bf16 and quant, greedy so it is reproducible, judged blind by a model from a
DIFFERENT family (Qwen3.8-27B q4: 0.996x its own bf16 ppl, no shared
quantization artifacts to sympathize with).

DESIGN GUARDS
  - blind + position-balanced: every pair is judged twice (A/B and B/A).
    A "win" requires the judge to pick the same continuation in BOTH orders;
    disagreement-with-itself counts as a tie. This cancels the judge's
    position bias exactly, the same trap --cyclic guards in litbench.
  - control pair: judge bf16 against ITSELF (self-consistency run) to read
    the judge's noise floor before trusting any verdict. If bf16-vs-bf16
    comes back far from 50/50 decisive, the judge is broken; expect ~all
    ties (identical text) — the real control is TWO DIFFERENT SEEDS of the
    prompt set, or simply reading `decisive` on the quant runs against it.
  - greedy generation: temp 0, so every number here is re-runnable.
  - sign test on decisive pairs: reported p is exact binomial, two-sided.

USAGE
  ./winrate_bench.py prompts   --out winrate/prompts.json
  ./winrate_bench.py generate  --model <dir> --prompts winrate/prompts.json \
      --out winrate/gens_<name>.json
  ./winrate_bench.py judge     --judge <dir> --a winrate/gens_bf16.json \
      --b winrate/gens_vqK2048.json --out winrate/verdict_K2048.json
a/b order does not matter (both orders are always judged); by convention
put bf16 as --a so "a_wins" reads as "bf16 wins".
"""
import argparse
import json
import math
import pathlib
import random
import re

HERE = pathlib.Path(__file__).parent

# ---------------------------------------------------------------- prompts

PROMPT_TASKS = [
    "Continue this passage for roughly 200 words, matching its voice, "
    "register, and period exactly. Output only the continuation.",
    "Continue this passage for roughly 200 words. Preserve the narrator's "
    "attitude and the scene's tension. Output only the continuation.",
]


def cmd_prompts(args):
    """Slice ~N passages out of the literary corpus, deterministically."""
    text = pathlib.Path(args.corpus).read_text()
    paras = [p.strip() for p in text.split("\n\n") if len(p.split()) > 40]
    rng = random.Random(args.seed)
    # take evenly spaced windows of 2-3 paragraphs so prompts span the corpus
    step = max(1, len(paras) // args.n)
    prompts = []
    for i in range(0, min(len(paras), step * args.n), step):
        chunk = "\n\n".join(paras[i:i + rng.choice([2, 3])])
        words = chunk.split()
        if len(words) > 260:
            chunk = " ".join(words[:260])
        task = PROMPT_TASKS[len(prompts) % len(PROMPT_TASKS)]
        prompts.append({"id": len(prompts), "passage": chunk, "task": task})
        if len(prompts) == args.n:
            break
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(prompts, indent=1))
    print(f"{len(prompts)} prompts -> {out}")


# --------------------------------------------------------------- generate

def strip_thinking(text):
    """Drop the model's thinking channel; keep only the shipped prose."""
    tail = text.rsplit("<channel|>", 1)[-1]
    return tail.strip()


def cmd_generate(args):
    import mlx.core as mx
    from mlx_lm import generate as mlx_generate
    from mlx_lm.utils import load, load_model, load_tokenizer
    try:
        model, tok, _ = load(args.model, return_config=True)
    except ValueError as e:
        if "not in model" not in str(e) or not args.allow_unmatched:
            raise
        p = pathlib.Path(args.model)
        model, _ = load_model(p, strict=False)
        tok = load_tokenizer(p)

    prompts = json.load(open(args.prompts))
    gens = []
    for pr in prompts:
        msg = [{"role": "user",
                "content": pr["passage"] + "\n\n" + pr["task"]}]
        text = tok.apply_chat_template(msg, add_generation_prompt=True,
                                       tokenize=False)
        out = mlx_generate(model, tok, prompt=text,
                           max_tokens=args.max_tokens, verbose=False)
        gens.append({"id": pr["id"], "text": strip_thinking(out)})
        mx.clear_cache()
        print(f"[{pr['id'] + 1}/{len(prompts)}] {len(gens[-1]['text'])} chars",
              flush=True)
    res = {"model": str(args.model).rstrip("/").split("/")[-1],
           "prompts": args.prompts, "max_tokens": args.max_tokens,
           "gens": gens}
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out}")


# ------------------------------------------------------------------ judge

JUDGE_TMPL = """You are judging two anonymous continuations of the same literary passage. Judge ONLY the quality of the writing: fidelity to the passage's voice and register, coherence, and literary craft. Ignore length differences.

PASSAGE:
{passage}

CONTINUATION 1:
{c1}

CONTINUATION 2:
{c2}

Which continuation is the better piece of literary writing? Answer with exactly one word: "1", "2", or "tie"."""


def parse_verdict(text):
    tail = text.rsplit("<channel|>", 1)[-1]
    for chunk in (tail, text):
        m = re.search(r"\b(1|2|tie)\b", chunk, re.IGNORECASE)
        if m:
            return m.group(1).lower()
    return None


def sign_test_p(wins, losses):
    """Exact two-sided binomial p on decisive pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def cmd_judge(args):
    import mlx.core as mx
    from mlx_lm import generate as mlx_generate
    from mlx_lm.utils import load
    judge, tok, _ = load(args.judge, return_config=True)

    A, B = json.load(open(args.a)), json.load(open(args.b))
    prompts = {p["id"]: p for p in json.load(open(A["prompts"]))}
    gens_a = {g["id"]: g["text"] for g in A["gens"]}
    gens_b = {g["id"]: g["text"] for g in B["gens"]}
    ids = sorted(set(gens_a) & set(gens_b))

    def ask(passage, c1, c2):
        msg = [{"role": "user", "content": JUDGE_TMPL.format(
            passage=passage, c1=c1, c2=c2)}]
        text = tok.apply_chat_template(msg, add_generation_prompt=True,
                                       tokenize=False)
        out = mlx_generate(judge, tok, prompt=text,
                           max_tokens=args.max_tokens, verbose=False)
        mx.clear_cache()
        return parse_verdict(out)

    a_wins = b_wins = ties = inconsistent = unparsed = 0
    detail = []
    for i in ids:
        pa, ca, cb = prompts[i]["passage"], gens_a[i], gens_b[i]
        v1 = ask(pa, ca, cb)            # A first
        v2 = ask(pa, cb, ca)            # B first — position-balanced
        if v1 is None or v2 is None:
            unparsed += 1
            verdict = "unparsed"
        elif v1 == "1" and v2 == "2":
            a_wins += 1; verdict = "a"
        elif v1 == "2" and v2 == "1":
            b_wins += 1; verdict = "b"
        elif v1 == "tie" and v2 == "tie":
            ties += 1; verdict = "tie"
        else:
            inconsistent += 1; verdict = "inconsistent"   # counts as a tie
        detail.append({"id": i, "o1": v1, "o2": v2, "verdict": verdict})
        print(f"[{i + 1}/{len(ids)}] {verdict}", flush=True)

    decisive = a_wins + b_wins
    res = {
        "a": A["model"], "b": B["model"], "judge":
            str(args.judge).rstrip("/").split("/")[-1],
        "n": len(ids), "a_wins": a_wins, "b_wins": b_wins, "ties": ties,
        "inconsistent": inconsistent, "unparsed": unparsed,
        "decisive": decisive,
        "a_winrate_decisive": round(a_wins / decisive, 4) if decisive else None,
        "sign_test_p": round(sign_test_p(a_wins, b_wins), 4),
        "detail": detail,
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"\n{A['model']} vs {B['model']}: "
          f"{a_wins}-{b_wins} ({ties} tie, {inconsistent} inconsistent, "
          f"{unparsed} unparsed) p={res['sign_test_p']}")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prompts")
    p.add_argument("--corpus",
                   default=str(HERE / "referee/referee_corpus_literary.txt"))
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_prompts)

    p = sub.add_parser("generate")
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=420)
    p.add_argument("--allow-unmatched", action="store_true")
    p.set_defaults(fn=cmd_generate)

    p = sub.add_parser("judge")
    p.add_argument("--judge", required=True)
    p.add_argument("--a", required=True, help="by convention: bf16")
    p.add_argument("--b", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=380)
    p.set_defaults(fn=cmd_judge)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
