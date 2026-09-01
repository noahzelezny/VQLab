# LinkedIn drafts — pick one, edit freely
# Canonical link for all three:
# https://thedrainflorist.com/ai/papers/data-free-vector-quantization/

---
## Variant A — story-led

I went to music school. Last night I published a quantization paper.

Over two weeks, my Macs and I built and measured vector-quantized versions
of three large language models — including a 397-billion-parameter model
that now runs on a single 128 GB Mac Studio.

The headline: below about 5 bits per weight, codebooks fit by plain k-means
over the weights — no calibration data, no teacher model — beat the affine
quantizations the community ships, at the same or smaller file sizes. Above
~5 bits, affine wins. The paper says exactly where, because we measured the
crossover instead of hiding it.

Everything is free and public: 13 models you can run today, the code
(Apache-2.0), and every negative result and corrected error, logged rather
than buried. A noise floor under every margin. No claim without a
measurement.

Paper: https://thedrainflorist.com/ai/papers/data-free-vector-quantization/
Models: https://huggingface.co/TheDrainFlorist
DOI: 10.5281/zenodo.22121193

---
## Variant B — results-led, short

New paper: Data-Free Vector Quantization Beats Affine Quantization at
Matched Bytes Below 6 Bits.

- A 397B-parameter LLM, running on one 128 GB Mac at ~20 tokens/sec
- Below ~5 bits/weight, data-free VQ (k-means over the weights, no
  calibration corpus, no teacher) beats the community's affine builds at
  matched or smaller sizes
- The advantage has a measured boundary: affine takes over at 4.5–6 bpw —
  we report where our method stops working, not just where it wins
- 13 free models, Apache-2.0 code, every number traceable to a lab record

Built and measured entirely on Apple Silicon with MLX, in two weeks.

Paper: https://thedrainflorist.com/ai/papers/data-free-vector-quantization/
Models: https://huggingface.co/TheDrainFlorist
DOI: 10.5281/zenodo.22121193

---
## Variant C — process-led (the honesty angle)

The most useful thing in the paper I just published might be the errors.

Seventeen of them — found by audits, logged in the record, corrected in
public. A title we weakened because one word in it wasn't supported. A
comparison we retracted because the noise floor said it was a tie. Negative
results with their own section.

The result that survived all that: vector quantization with zero calibration
data beats the affine quantizations people actually ship, below about 5 bits
per weight, on three models from 27B to 397B parameters — measured, with a
noise floor under every margin, on Apple Silicon.

I came to this from music school, not a PhD. What I had instead of
credentials was a rule: no claim without a measurement, and no measurement
without knowing its noise.

13 free models. Code is Apache-2.0. Every number traces.

Paper: https://thedrainflorist.com/ai/papers/data-free-vector-quantization/
DOI: 10.5281/zenodo.22121193
