"""Serve a VQLab artifact over an OpenAI-compatible API, with MTP drafting.

    python -m vqlab.cli serve --model <artifact> [--sidecar <head>]
        [--host 127.0.0.1] [--port 8080]

This is an ADAPTER, not a server. mlx-lm already ships a complete
OpenAI-compatible server -- templates, streaming, stop sequences, the request
schema, model loading -- and it calls generation at exactly one site. So
VQLab does not stand in for that surface; it borrows it and replaces the
decode strategy underneath, which is the same seam the rest of this package
uses. Roughly: mlx-lm serves, VQLab decodes.

Why serve at all, when VQLab is a research tool: a VQ artifact needs the
codebook kernels in its own bundled model.py, so the environment that runs it
is not interchangeable. Shipping the server that we know loads our artifacts
is the difference between a downloader getting a working model and getting a
stack trace. The MTP head is a bonus on top of that, not the reason.

Four patches, all narrow, all reversible, and each one is a place mlx-lm could
move underneath us:

  make_sampler       wrapped so the sampling PARAMETERS travel with the
                     sampler. mlx-lm's sampler returns a token; exact
                     rejection sampling needs the distribution the token came
                     from. Without the params we could only fall back to
                     accept-if-equal, which silently drops the guarantee that
                     temperature output matches plain sampling -- so we
                     recover the params rather than degrade quietly.
  stream_generate    replaced with the MTP loop, yielding mlx-lm's own
                     GenerationResponse so the server is none the wiser.
  _is_batchable      forced False, backing up --decode-concurrency 1. The
                     MTP loop is single-sequence; batching it is a real piece
                     of work (see MTPLX's continuous-batch MTP handling), not
                     a flag.
  fetch_nearest_cache returns a miss, backing up --prompt-cache-size 0.
                     Cross-request prefix reuse would hand us a trunk cache
                     already holding N positions while the head is seeded
                     from position 0 -- reintroducing exactly the
                     misalignment that cost 5.9pp of acceptance. Pairing a
                     head cache with each cached prefix is the fix; until
                     then reuse is off and multi-turn chats re-prefill.
                     The loop RAISES on a non-empty cache rather than
                     decoding at wrong positions, so if these ever stop
                     working it fails loudly.

Without --sidecar this still serves the artifact, just without drafting, which
is the useful default for a model whose runtime is the hard part.
"""
import argparse
import logging
import sys

import mlx.core as mx

_HEAD = {}


def _wrap_make_sampler(real):
    """Keep the sampling parameters attached to the sampler mlx-lm builds."""
    def make_sampler(temp=0.0, top_p=0.0, min_p=0.0, min_tokens_to_keep=1,
                     top_k=0, xtc_probability=0.0, xtc_threshold=0.0,
                     xtc_special_tokens=[], **kw):
        fn = real(temp=temp, top_p=top_p, min_p=min_p,
                  min_tokens_to_keep=min_tokens_to_keep, top_k=top_k,
                  xtc_probability=xtc_probability,
                  xtc_threshold=xtc_threshold,
                  xtc_special_tokens=xtc_special_tokens, **kw)
        try:
            fn.vqlab_params = dict(
                temp=temp, top_p=top_p, min_p=min_p,
                min_tokens_to_keep=min_tokens_to_keep, top_k=top_k,
                xtc_probability=xtc_probability, xtc_threshold=xtc_threshold,
                xtc_special_tokens=xtc_special_tokens)
        except AttributeError:      # a builtin or C callable: fall back
            pass
        return fn
    make_sampler.vqlab_patched = True
    return make_sampler


def _make_stream_generate(real_stream_generate):
    """Route through the MTP loop when a head is loaded; else stock mlx-lm."""
    from mlx_lm.generate import GenerationResponse

    from vqlab.mtp import mtp_stream_generate

    def stream_generate(model=None, tokenizer=None, prompt=None,
                        max_tokens=256, sampler=None, logits_processors=None,
                        prompt_cache=None, draft_model=None,
                        num_draft_tokens=None, prompt_progress_callback=None,
                        prefill_step_size=2048, **kw):
        head = _HEAD.get("head")
        if head is None or draft_model is not None:
            # No head, or the operator asked for stock draft-model
            # speculation: hand the request straight back to mlx-lm.
            return real_stream_generate(
                model=model, tokenizer=tokenizer, prompt=prompt,
                max_tokens=max_tokens, sampler=sampler,
                logits_processors=logits_processors, prompt_cache=prompt_cache,
                draft_model=draft_model, num_draft_tokens=num_draft_tokens,
                prompt_progress_callback=prompt_progress_callback,
                prefill_step_size=prefill_step_size, **kw)

        params = dict(getattr(sampler, "vqlab_params", None) or {})
        n_prompt = len(prompt) if hasattr(prompt, "__len__") else 0
        logging.info("MTP: drafting this request (%d prompt tokens, temp %s)",
                     n_prompt, params.get("temp", 0.0))

        def gen():
            import time
            t0 = time.perf_counter()
            first, prompt_tps, last = True, 0.0, None
            try:
                for r in mtp_stream_generate(
                        model, tokenizer, prompt, head, max_tokens=max_tokens,
                        logits_processors=logits_processors,
                        prefill_step_size=prefill_step_size,
                        prompt_cache=prompt_cache, want_logprobs=True,
                        **params):
                    if first:
                        dt = time.perf_counter() - t0
                        prompt_tps = (n_prompt / dt) if dt else 0.0
                        first = False
                    last = r
                    yield GenerationResponse(
                        text=r.text, token=r.token, logprobs=r.logprobs,
                        from_draft=r.from_draft, prompt_tokens=n_prompt,
                        prompt_tps=prompt_tps,
                        generation_tokens=r.generation_tokens,
                        generation_tps=r.generation_tps,
                        peak_memory=r.peak_memory,
                        finish_reason=r.finish_reason)
            finally:
                # In a finally so the summary survives the server breaking out
                # of the loop early on a stop condition -- and so an operator
                # can always see that drafting actually ran.
                if last is not None:
                    logging.info(
                        "MTP: acceptance %.3f over %d steps, %d tokens, "
                        "%.2f tok/s", last.acceptance, last.steps,
                        last.generation_tokens, last.generation_tps)

        return gen()

    stream_generate.vqlab_patched = True
    return stream_generate


def install(model_path, sidecar=None, family=None, quiet=False):
    """Patch mlx-lm's server in place. Returns the server module."""
    from mlx_lm import server as srv

    srv.make_sampler = _wrap_make_sampler(srv.make_sampler)
    srv.stream_generate = _make_stream_generate(srv.stream_generate)

    # Single-sequence only. This MUST be ResponseGenerator: mlx-lm decides
    # batching there (`_is_batchable`), and the batch path runs BatchGenerator
    # which never calls stream_generate at all -- so patching the wrong class
    # leaves the server working perfectly while silently not drafting.
    # qwen4_exp's _AttnCache defines merge(), so batching IS eligible here and
    # this is load-bearing, not belt-and-braces.
    if not hasattr(srv, "ResponseGenerator"):
        raise RuntimeError(
            "mlx_lm.server has no ResponseGenerator: the batching decision has "
            "moved and VQLab cannot guarantee the MTP loop is used. Refusing "
            "to serve rather than silently decode without drafting.")
    srv.ResponseGenerator._is_batchable = lambda self, args: False

    # No cross-request prefix reuse while the head cannot follow a reused
    # trunk cache. Reported as a miss so the server builds a fresh one.
    try:
        from mlx_lm.models.cache import LRUPromptCache
        LRUPromptCache.fetch_nearest_cache = lambda self, key, prompt: (None,
                                                                        prompt)
    except Exception as e:                                   # noqa: BLE001
        print(f"warning: could not disable prompt-cache reuse ({e}); the MTP "
              f"loop will refuse requests rather than mis-position the head",
              file=sys.stderr)
    return srv


def selfcheck(srv):
    """Refuse to start if any patch point has moved.

    A server that quietly stops drafting still answers every request
    correctly, just slower -- the failure is invisible from the outside, which
    is exactly why it gets checked here instead of being discovered in a
    benchmark weeks later."""
    problems = []
    if getattr(srv.stream_generate, "__name__", "") != "stream_generate" or \
            not getattr(srv.stream_generate, "vqlab_patched", False):
        problems.append("stream_generate is not the VQLab wrapper")
    if srv.ResponseGenerator._is_batchable(object(), None) is not False:
        problems.append("batching is still enabled")
    if not getattr(srv.make_sampler, "vqlab_patched", False):
        problems.append("make_sampler is not wrapped (temperature would lose "
                        "its exactness guarantee)")
    if problems:
        raise RuntimeError("vqlab serve self-check failed: "
                           + "; ".join(problems))
    return True


def main():
    ap = argparse.ArgumentParser(
        description="serve a VQLab artifact over an OpenAI-compatible API")
    ap.add_argument("--model", required=True)
    ap.add_argument("--sidecar", default=None,
                    help="MTP head; without it the artifact serves normally")
    ap.add_argument("--family", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    a, rest = ap.parse_known_args()

    srv = install(a.model, a.sidecar, a.family)
    selfcheck(srv)
    print('vqlab serve: patch self-check passed', flush=True)

    argv = [sys.argv[0], "--model", a.model, "--host", a.host,
            "--port", str(a.port), "--trust-remote-code"]
    if a.sidecar:
        # Supported flags first; the monkeypatches above are the backstop.
        argv += ["--prompt-cache-size", "0", "--decode-concurrency", "1"]
    argv += rest
    sys.argv = argv
    if a.sidecar:
        # The server loads its own copy of the model; bind the head to it the
        # first time a request arrives, so we never hold two trunks resident.
        real_load = srv.ModelProvider.load

        def load(self, *args, **kwargs):
            out = real_load(self, *args, **kwargs)
            if _HEAD.get("head") is None:
                from vqlab.mtp import load_mtp_head
                before = mx.get_active_memory()
                head, spec = load_mtp_head(self.model, sidecar=a.sidecar,
                                           model_path=a.model,
                                           family=a.family)
                mx.clear_cache()
                _HEAD["head"] = head
                print(f"MTP head ({spec.name}) bound: "
                      f"{(mx.get_active_memory() - before) / 2**30:.2f} GiB "
                      f"resident; drafting enabled", flush=True)
            return out

        srv.ModelProvider.load = load
    else:
        print("no --sidecar: serving without MTP drafting", flush=True)
    return srv.main()


if __name__ == "__main__":
    sys.exit(main())
