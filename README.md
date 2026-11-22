# Abliterating DeepSeek-V4-Flash-0731

This repository removes refusal behaviour from
[DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
(284B parameters, 13B active) by directional ablation ("abliteration"): it
measures the direction in the model's activations that separates requests it
refuses from ones it answers, and projects that direction out of the weights
that write to the residual stream. It contains all the code that produced the
build and a full account of the run, including what failed.

The work was done on two NVIDIA GB10 nodes (DGX Spark class, 128 GB unified
memory each) joined by a 200 Gb/s link, serving the model with vLLM at tensor
parallel 2. No weights are distributed here.

## Result

Stock model against the kept build (experiment `m2.5-13-35`). The judge is the
unmodified model itself, and held-out prompts were never used to choose the
configuration.

| | stock | abliterated |
|---|---|---|
| 300 held-out harmful prompts, thinking off: comply / deflect / refuse | 7% / 11% / 81% | **84% / 13% / 0%** |
| same, thinking on | 15% / 8% / 77% | **89% / 7% / 0%** |
| 100 benign prompts that sound harmful (JailbreakBench benign): answered | 79% | **97%** |
| MMLU, 500 questions | 402 | **402** |
| GSM8K, 200 problems, thinking off / on | 194 / 188 | **195 / 190** |
| first-token KL from stock on harmless prompts (mean) | 0 | 0.28 |
| decode speed, one stream (with the DSpark draft head) | 44.5 tok/s | 45.9 tok/s |

Refusals are gone in both reasoning modes. What remains is deflection: answering
a safer version of the request. Capability and speed are unchanged. A cautious
system prompt brings back some reluctance (63% comply). Full tables, including
every rung that lost, are in [docs/results.md](docs/results.md).

## Why this model was hard

Abliteration is routine on a dense BF16 model that fits on one GPU. Four things
here broke the usual recipe.

1. **There is no BF16 original.** DeepSeek released V4 in the precision it was
   trained in: FP8 (E4M3, 128×128 blocks with power-of-two scales) for attention
   and the shared expert, FP4 (E2M1, a power-of-two scale per 32 inputs) for the
   256 routed experts. Every edited matrix is decoded, edited in FP32 and
   re-encoded in the same format. The encoders reproduce the stock checkpoint's
   values exactly. [docs/quantization.md](docs/quantization.md)
2. **Rounding undid the edit.** The stock weights sit exactly on the storage
   grid, and an edit moves each element by a fraction of a grid step. Rounding to
   nearest puts most elements back, and with them most of the removed direction:
   89-92% of it in the FP4 experts. The fix is direction-aware rounding: elements
   are moved to their other neighbouring grid value, cheapest first, until the
   stored matrix has exactly the edit's component along the direction. Every
   build checks that the edit survived. This idea came from
   [arXiv 2611.09382](https://arxiv.org/abs/2611.09382)'s warning that weight
   edits can fall below storage resolution.
3. **Hyper-connections.** The residual stream is four copies of the hidden state,
   mixed by learned matrices. Directions are measured on the mean of the copies,
   which is the reduction DeepSeek's own draft head reads. The four copies'
   individual refusal directions disagree, but removing one direction turned out
   to be enough. [docs/architecture.md](docs/architecture.md)
4. **It fills both nodes.** Every candidate is a 156 GiB build that must be
   copied to the second node and served across both, so nothing can run beside
   it, including a judge. Candidates only collect responses, and the stock model
   judges them once it is served again.

Two findings decided the result.

- **The last seven layers break the model.** The usual heuristic put the edit in
  layers 20-42. That made the model loop ("I I I I…") and drift heavily, with
  first-token KL of 2.4-4.0. A lab screen that applies edits exactly, without
  serving, showed that layers 36-42 cause it. Stopping at layer 35 cut drift 25×.
- **Thinking mode needs a stronger edit.** With reasoning off, a modest edit
  removed refusals. With reasoning on, the model deliberated its way back to
  refusing until the edit was strengthened (weight 2.5) and started earlier
  (layer 13).

## How it works

[docs/method.md](docs/method.md) covers the method in full.

1. **Directions** ([`lab/stream_dsv4.py`](lab/stream_dsv4.py)): run 400 harmful
   and 400 harmless prompts through the model one layer at a time (it does not
   fit in memory whole) and take the difference of mean last-token activations
   at every layer. The streaming implementation is gated against the served
   model's first-token distributions.
2. **Edit** ([`tools/ablate.py`](tools/ablate.py)): for each layer in a band,
   `W' = W − w·v vᵀW` for every matrix that writes to the residual stream:
   attention's output projection `wo_b`, the shared expert's `w2`, and all 256
   routed experts' `w2`. That is 5,934 matrices for layers 13-35.
3. **Screen** ([`abliterate.sh`](abliterate.sh)): build, serve and query a ladder
   of strengths and layer bands, plus an unedited control.
4. **Judge** ([`tools/judge.py`](tools/judge.py)): the stock model classifies
   each response as COMPLY, PARTIAL, DEFLECT or REFUSE.
   [docs/evaluation.md](docs/evaluation.md)
5. **Choose and validate**: the builds within 3 points of the best compliance
   rate are finalists, and the least drift breaks ties. Finalists face held-out
   prompts, benign prompts, MMLU, GSM8K, robustness scenarios and a speed check
   against fixed limits.

## Layout

| | |
|---|---|
| [`abliterate.sh`](abliterate.sh) | the whole pipeline, resumable, as stages |
| [`queue.sh`](queue.sh), [`box/run_queue.sh`](box/run_queue.sh) | build, copy, serve and collect a queue of experiments (runs detached on the head node) |
| [`box/judge_pending.sh`](box/judge_pending.sh) | judges collected responses once the stock model is back |
| [`lab.sh`](lab.sh), [`lab/`](lab/) | the streaming lab: directions, the exact screen, the gate |
| [`finalize.sh`](finalize.sh) | rebuilds the winner under its kept name, with `ABLATION.txt` |
| [`tools/`](tools/) | formats, the edit, evaluation, judging, selection |
| [`models/deepseek-v4-flash-0731.env`](models/deepseek-v4-flash-0731.env) | every model-specific setting of the run |
| [`serving/`](serving/) | the vLLM profiles and launch command; a reference switch script |
| [`data/`](data/) | benign and capability sets; the harmful sets are rebuilt by script |
| [`results/ABLATION.txt`](results/ABLATION.txt) | the record written into the kept checkpoint |
| [`tests/`](tests/) | the formats and the rounding fix, on synthetic matrices |
| [`docs/`](docs/) | architecture, method, quantization, evaluation, results, reproducing, lessons |

## Reproducing

See [docs/reproducing.md](docs/reproducing.md). In short:

```bash
cp site.env.example site.env           # your nodes and serving arrangement
python3 tools/build_prompt_sets.py data/prompts   # rebuilds the harmful sets (checksums in data/SHA256SUMS)
bash abliterate.sh plan                # what it will do
bash abliterate.sh run                 # about 16 hours, most of it with the cluster serving test builds
```

`python3 tests/test_quant_formats.py` runs without the model.

## Responsible use

An abliterated model does not refuse. It will produce content the original
would decline: dangerous instructions, harassment, material that is illegal
where you live. This repository exists to document a technique and its
measurement on a frontier-scale open-weight model, for research and for
private, accountable use. If you build and run such a model, you are
responsible for what it is used for. Keep it out of products and services
that reach people who did not choose an uncensored model. Follow the model's
licence and your local law. No weights, and no model outputs to the harmful
prompts, are published here. The repository keeps aggregate numbers only.

## Credits

- [Heretic](https://github.com/p-e-w/heretic): this project began as tooling
  around it. The refusal-keyword list in `tools/eval_refusals.py` is Heretic's.
- DeepSeek's reference inference code (`inference/kernel.py`) defined the scale
  rule the encoders follow. The checkpoint's own `encoding/encoding_dsv4.py`
  renders the lab's prompts.
- Hugging Face transformers' `DeepseekV4Model` is what the lab runs, layer by
  layer.
- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) launched
  vLLM across the two nodes.
- The prompt and capability sets come from public benchmarks, listed in
  [data/README.md](data/README.md).
- Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction*
  (2024), for the method.

## License

AGPL-3.0 (see [LICENSE](LICENSE)), the licence of Heretic, from which the
keyword list is taken.
