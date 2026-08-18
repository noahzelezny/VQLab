#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""litbench scored CHAT-NATIVELY — the only fair way to compare gemma-4.

WHY THIS EXISTS. litbench as shipped scores each choice as a raw continuation
(lm-eval multiple_choice). That is invalid on gemma-4: the family's raw-text
distribution is collapsed, so raw-continuation scoring put a 26B MoE BELOW
4-choice chance (21.2%) while it generates fluent prose (E39,
GEMMA4_PPL_ANOMALY.md). Comparing gemma to Qwen that way measures the
instrument, not the models.

Chat-tuned models are IN-DISTRIBUTION inside their own chat template
answering a lettered multiple-choice question. So: present the passage and
question through each model's OWN template, label the options A-D, and score
the single answer-letter token. Every model is then measured in the regime
it was trained for, which is also the regime the sidecar actually runs in.

WHY THIS IS ALSO CHEAPER. One forward pass per item (read four letter
logprobs off the final position) instead of one per choice. 4x less compute
than the continuation form.

WHAT IT DOES NOT FIX. Position bias is real in lettered MC: a model can
prefer "A" irrespective of content. Guard = --cyclic, which re-presents each
item with the options rotated through all 4 positions and averages. That is
4x the passes, so it is off by default and ON is the number to quote when
two models are being compared for a decision.

*** READ THIS BEFORE USING SINGLE-TOKEN MODE TO COMPARE TWO MODELS ***

Single-token scoring silently measures WILLINGNESS TO ANSWER IMMEDIATELY,
not comprehension, whenever a model is a reasoner. Measured on these two:
the top-5 next tokens after the generation prompt are

    e4b bf16 : '<|channel>', 'D', 'A', '$', 'C'      <- letters present
    26b bf16 : '<|channel>', '<', '---', ' <', ' inner'  <- NO letter at all

Both want to open a thinking channel; the 26b is simply more committed to
it, and writes a real analysis of each option before answering. Reading
letter logprobs at that position therefore penalises the better reasoner.
It produced e4b 78.85% vs 26b 37.5% — a gap that is mostly ARTEFACT.

Use --generative for any cross-model comparison: it lets the model think,
then parses the answer letter out of the completion. Slower, and correct.
Single-token mode remains useful for comparing QUANTS OF ONE MODEL, where
both sides share the same answering style and the bias cancels.

    ./litbench_chat.py --model <dir> --out results_literary/<name>.json
    ./litbench_chat.py --model <dir> --cyclic      # position-debiased
"""
import argparse
import json
import pathlib

import mlx.core as mx

LETTERS = ["A", "B", "C", "D"]
HERE = pathlib.Path(__file__).parent


def build_prompt(tok, item, rot):
    """Rotate options by `rot`, return (chat text, index of gold letter)."""
    n = len(item["choices"])
    order = [(i + rot) % n for i in range(n)]
    lines = [item["passage"], "", item["question"], ""]
    for pos, src in enumerate(order):
        lines.append(f"{LETTERS[pos]}. {item['choices'][src]}")
    lines.append("")
    lines.append("Answer with a single letter.")
    msg = [{"role": "user", "content": "\n".join(lines)}]
    text = tok.apply_chat_template(msg, add_generation_prompt=True,
                                   tokenize=False)
    gold_pos = order.index(item["label"])
    return text, gold_pos


def letter_ids(tok):
    """Token id for each letter as it appears at the start of an answer.

    Encoded WITHOUT special tokens and taking the last piece, so this works
    whether the tokenizer emits 'A' or ' A' after the generation prompt.
    """
    ids = {}
    for i, L in enumerate(LETTERS):
        cand = tok.encode(L, add_special_tokens=False)
        ids[i] = cand[-1]
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", default=str(HERE / "literary" / "litbench.jsonl"))
    ap.add_argument("--cyclic", action="store_true",
                    help="rotate options through all 4 positions and average "
                         "(4x cost, removes position bias)")
    ap.add_argument("--generative", action="store_true",
                    help="let the model think, then parse the answer letter. "
                         "REQUIRED for cross-model comparison (see header).")
    ap.add_argument("--max-tokens", type=int, default=640,
                    help="generation budget in --generative mode; the 26b "
                         "spends 200+ tokens reasoning before answering")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-unmatched", action="store_true")
    args = ap.parse_args()

    from mlx_lm.utils import load, load_model, load_tokenizer
    try:
        model, tok, _ = load(args.model, return_config=True)
    except ValueError as e:
        if "not in model" not in str(e) or not args.allow_unmatched:
            raise
        p = pathlib.Path(args.model)
        model, _ = load_model(p, strict=False)
        tok = load_tokenizer(p)

    items = [json.loads(l) for l in open(args.items) if l.strip()]
    if args.limit:
        items = items[:args.limit]
    lids = letter_ids(tok)
    rots = range(len(LETTERS)) if args.cyclic else [0]

    from mlx_lm import generate as mlx_generate
    import re as _re

    def parse_letter(text):
        """First standalone A-D AFTER the thinking channel closes.

        The template wraps reasoning as <|channel>thought ... <channel|>, so
        prefer text after the close; fall back to the whole completion.
        """
        tail = text.rsplit("<channel|>", 1)[-1]
        for chunk in (tail, text):
            m = _re.search(r"\b([ABCD])\b", chunk)
            if m:
                return "ABCD".index(m.group(1))
        return None

    correct = 0
    per_cat = {}
    picks = [0] * len(LETTERS)
    unparsed = 0
    per_item = []          # per-item outcomes -> paired tests (McNemar) later
    for it in items:
        votes = [0.0] * len(LETTERS)
        gen_votes = [0] * len(LETTERS)
        for rot in rots:
            text, gold_pos = build_prompt(tok, it, rot)
            if args.generative:
                out = mlx_generate(model, tok, prompt=text,
                                   max_tokens=args.max_tokens, verbose=False)
                pos = parse_letter(out)
                if pos is None:
                    unparsed += 1
                else:
                    n = len(it["choices"])
                    gen_votes[(pos + rot) % n] += 1
                mx.clear_cache()
                continue
            ids = tok.encode(text)
            logits = model(mx.array([ids])).astype(mx.float32)[0, -1]
            lse = mx.logsumexp(logits)
            lp = [float((logits[lids[i]] - lse).item())
                  for i in range(len(LETTERS))]
            mx.eval(logits)
            # map each presented position back to the ORIGINAL choice index
            n = len(it["choices"])
            for pos in range(n):
                src = (pos + rot) % n
                votes[src] += lp[pos]
            del logits
            mx.clear_cache()
        src_votes = gen_votes if args.generative else votes
        pred = max(range(len(src_votes)), key=lambda i: src_votes[i])
        picks[pred] += 1
        ok = int(pred == it["label"])
        correct += ok
        c = per_cat.setdefault(it["category"], [0, 0])
        c[0] += ok
        c[1] += 1
        per_item.append({"id": it.get("id", len(per_item)),
                         "category": it["category"], "pred": pred,
                         "gold": it["label"], "correct": ok})

    acc = correct / len(items)
    res = {
        "model": str(args.model).rstrip("/").split("/")[-1],
        "n": len(items), "accuracy": round(acc, 4),
        "cyclic": args.cyclic,
        "chance": round(1 / len(LETTERS), 4),
        "per_category": {k: round(v[0] / v[1], 4) for k, v in sorted(per_cat.items())},
        "answer_distribution": picks,
        "mode": "generative" if args.generative else "single_token",
        "unparsed": unparsed,
        "per_item": per_item,
    }
    print(json.dumps(res, indent=1))
    print(f"\naccuracy {acc:.2%}  (chance {1/len(LETTERS):.0%}, n={len(items)}"
          f"{', position-debiased' if args.cyclic else ''})")
    if not args.cyclic:
        print(f"answer distribution {picks} — - skew here means position bias; "
              f"re-run with --cyclic before quoting a comparison.")
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(res, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
