"""The MTP speculative decode loop.

This is a decode *strategy*, not a runtime. mlx-lm loads the model and owns
the architecture; the only contract we depend on is

    model(tokens, cache=cache) -> logits          and    model.make_cache()

plus one per-family capture point for the pre-lm_head activation (capture.py).
Everything else here is the loop.

The head drafts token t+2 from (trunk hidden at t, embedding of t+1), so each
step verifies exactly one speculative token inside a single 2-token trunk
forward:

  accepted -> two tokens for one trunk forward.
  rejected -> the trunk's own t+2 came out of that same forward, so the step
              still emits two tokens; the caches roll back and the accepted
              pair is replayed so the state matches what was emitted.

A rejection therefore costs one extra forward, never a wrong token. At
temperature the acceptance test is exact rejection sampling with residual
correction (sampling.rejection_correct), so the emitted sequence is
distributed exactly as plain sampling from the trunk would be, for any draft
quality. At temperature 0 that rule degenerates to `draft == argmax`, which is
the greedy branch below.

What this does NOT give is bit-identical output against single-token decoding,
and demanding it is a false gate: MLX's chunked and single-token kernels
disagree at genuine near-ties (measured top-2 logit gaps of 0.25 and exactly
0.00 against a median of 3.625), and verification always happens inside a
2-token forward. Every correct speculative implementation on this runtime
inherits that. The gate is "divergence confined to near-ties" — `vqlab
mtp-bench` measures it directly.

Known and deliberate: the head's own KV cache advances one position per step
while two tokens commit, so it is only approximately aligned. That costs
acceptance, never correctness — the trunk verifies every drafted token.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator, List, Optional, Union

import mlx.core as mx

from . import registry
from .capture import capture_input
from .caches import restore, snapshot
from .sampling import Distribution, make_distribution, rejection_correct

__all__ = ["MTPResponse", "load_mtp_head", "mtp_generate", "mtp_stream_generate"]


@dataclass
class MTPResponse:
    """One emitted token, plus the running speculative statistics."""
    text: str
    token: int
    from_draft: bool          # emitted as an ACCEPTED draft (a free token)
    finish_reason: Optional[str]
    steps: int
    accepted: int
    acceptance: float         # accepted / steps
    generation_tokens: int
    generation_tps: float
    peak_memory: float
    tail: bool = False        # trailing detokenizer flush, not a new token


def load_mtp_head(model, sidecar=None, family: Optional[str] = None,
                  model_path=None):
    """Load a drafting head for `model`.

    `sidecar` may be a file; if omitted, the family's sidecar name is looked
    for in `model_path`. Returns (head, spec).

    The head is optional by construction: sidecars are named outside mlx-lm's
    `model*.safetensors` glob, so a model directory carrying one still loads
    normally through the stock loader.
    """
    import pathlib

    spec = registry.resolve(model, family)
    if sidecar is None:
        if model_path is None:
            raise ValueError("pass either sidecar= or model_path=")
        sidecar = pathlib.Path(model_path) / spec.sidecar_name
    sidecar = pathlib.Path(sidecar)
    if not sidecar.exists():
        raise FileNotFoundError(
            f"no MTP sidecar at {sidecar}; build one with `vqlab mtp-pack`. "
            f"The head is optional — without it the model decodes normally.")
    arch = spec.arch_module(model)
    return spec.head_cls().from_sidecar(model, arch, sidecar), spec


def _encode(tokenizer, prompt) -> mx.array:
    if isinstance(prompt, mx.array):
        ids = prompt
    elif isinstance(prompt, str):
        ids = mx.array(tokenizer.encode(prompt))
    else:
        ids = mx.array(list(prompt))
    if ids.ndim == 1:
        ids = ids[None]
    if ids.shape[0] != 1:
        raise ValueError("speculative decoding here is single-sequence; "
                         f"got a batch of {ids.shape[0]}")
    return ids


def _eos_ids(tokenizer) -> set:
    ids = getattr(tokenizer, "eos_token_ids", None)
    if ids:
        return set(ids)
    tid = getattr(tokenizer, "eos_token_id", None)
    return {tid} if tid is not None else set()


class _Detok:
    """mlx-lm's streaming detokenizer when there is one, whole-text decode
    otherwise, so a bare HF tokenizer still works."""

    def __init__(self, tokenizer):
        self._d = getattr(tokenizer, "detokenizer", None)
        self._tok = tokenizer
        self._ids: List[int] = []
        if self._d is not None:
            self._d.reset()

    def add(self, token: int) -> str:
        if self._d is not None:
            self._d.add_token(token)
            return self._d.last_segment
        self._ids.append(token)
        prev = self._text if hasattr(self, "_text") else ""
        self._text = self._tok.decode(self._ids)
        return self._text[len(prev):]

    def finalize(self) -> str:
        if self._d is not None:
            self._d.finalize()
            return self._d.last_segment
        return ""


def mtp_stream_generate(
    model,
    tokenizer,
    prompt,
    head,
    *,
    family: Optional[str] = None,
    max_tokens: int = 256,
    temp: float = 0.0,
    top_p: float = 0.0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 1,
    top_k: int = 0,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.0,
    xtc_special_tokens: List[int] = [],
    logits_processors: Optional[List[Callable]] = None,
    prefill_step_size: int = 2048,
) -> Iterator[MTPResponse]:
    """Stream tokens from `model`, drafting each second token with `head`.

    Sampling parameters mirror `mlx_lm.sample_utils.make_sampler`; the
    resulting output distribution is the one mlx-lm would produce for the same
    parameters, preserved through verification by rejection sampling.
    """
    spec = registry.resolve(model, family)
    dist = make_distribution(temp, top_p, min_p, min_tokens_to_keep, top_k,
                             xtc_probability, xtc_threshold, xtc_special_tokens)
    copy_caches = spec.cache_semantics != "reassign"
    eos = _eos_ids(tokenizer)
    detok = _Detok(tokenizer)
    ids = _encode(tokenizer, prompt)
    arch = spec.arch_module(model)

    emitted: List[int] = []
    processors = list(logits_processors or [])

    def pick(row: mx.array):
        """row: [1, V] -> (token [1], Distribution or None)."""
        for proc in processors:
            row = proc(mx.array(emitted), row)
        if dist is None:
            return mx.argmax(row, axis=-1), None
        d = dist(row)
        return d.sample(), d

    with capture_input(model.model, spec.capture) as get_h:
        cache = model.make_cache()
        dcache = spec.make_draft_cache(arch)

        # prefill
        n = ids.shape[1]
        for i in range(0, n - 1, prefill_step_size):
            chunk = ids[:, i:min(i + prefill_step_size, n - 1)]
            if chunk.shape[1]:
                mx.eval(model(chunk, cache=cache))
                mx.clear_cache()
        logits = model(ids[:, max(n - 1, 0):], cache=cache)
        h_last = get_h()[:, -1:]
        t1, _ = pick(logits[:, -1])
        mx.eval(t1, h_last)

        steps = accepted = 0
        start = time.perf_counter()
        finish: Optional[str] = None

        while finish is None:
            steps += 1
            dsnap = snapshot([dcache], copy=copy_caches)
            draft_row = head.draft_logits(h_last, t1[None], dcache)[:, -1]
            if dist is None:
                d2 = mx.argmax(draft_row, axis=-1)
                q = None
            else:
                for proc in processors:
                    draft_row = proc(mx.array(emitted), draft_row)
                q = dist(draft_row)
                d2 = q.sample()
            mx.eval(d2)

            csnap = snapshot(cache, copy=copy_caches)
            lg2 = model(mx.concatenate([t1, d2])[None], cache=cache)

            if dist is None:
                true_t2 = mx.argmax(lg2[:, 0], axis=-1)
                mx.eval(true_t2)
                ok = bool((true_t2 == d2).item())
                t2 = d2 if ok else true_t2
            else:
                p = dist(lg2[:, 0])
                acc, t2 = rejection_correct(p.probs, q.probs, d2)
                mx.eval(acc, t2)
                ok = bool(acc.item())

            if ok:
                accepted += 1
            else:
                # Roll back to before the drafted pair and replay the tokens
                # actually emitted, so the caches match the emitted stream.
                restore(cache, csnap)
                restore([dcache], dsnap)
                lg2 = model(mx.concatenate([t1, t2])[None], cache=cache)

            for tok, from_draft in ((t1, False), (t2, ok)):
                token = int(tok.item())
                emitted.append(token)
                if token in eos:
                    finish = "stop"
                elif len(emitted) >= max_tokens:
                    finish = "length"
                elapsed = time.perf_counter() - start
                yield MTPResponse(
                    text=detok.add(token), token=token, from_draft=from_draft,
                    finish_reason=finish, steps=steps, accepted=accepted,
                    acceptance=accepted / steps, generation_tokens=len(emitted),
                    generation_tps=len(emitted) / elapsed if elapsed else 0.0,
                    peak_memory=mx.get_peak_memory() / 2**30,
                )
                if finish is not None:
                    break
            if finish is not None:
                break

            h_last = get_h()[:, -1:]
            t1, _ = pick(lg2[:, 1])
            mx.eval(t1, h_last)

        tail = detok.finalize()
        if tail:
            elapsed = time.perf_counter() - start
            yield MTPResponse(
                text=tail, token=emitted[-1], from_draft=False, tail=True,
                finish_reason=finish, steps=steps, accepted=accepted,
                acceptance=accepted / max(steps, 1),
                generation_tokens=len(emitted),
                generation_tps=len(emitted) / elapsed if elapsed else 0.0,
                peak_memory=mx.get_peak_memory() / 2**30,
            )


def mtp_generate(model, tokenizer, prompt, head, *, verbose: bool = False,
                 **kwargs) -> str:
    """Generate a completion with MTP speculative drafting; return the text.

    Mirrors `mlx_lm.generate`'s shape. `mtp_stream_generate` exposes the
    per-token stream and the acceptance statistics.
    """
    parts, last = [], None
    for r in mtp_stream_generate(model, tokenizer, prompt, head, **kwargs):
        parts.append(r.text)
        if verbose:
            print(r.text, end="", flush=True)
        last = r
    if verbose and last is not None:
        print(f"\n\n{last.generation_tokens} tokens, "
              f"{last.generation_tps:.2f} tok/s, "
              f"acceptance {last.acceptance:.3f} over {last.steps} steps, "
              f"peak memory {last.peak_memory:.2f} GiB", flush=True)
    return "".join(parts)
