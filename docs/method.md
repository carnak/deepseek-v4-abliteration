# Method

A model's tendency to refuse shows up as a direction in its activations: the
difference between the mean activation on harmful and on harmless prompts
(Arditi et al., 2024). Every layer writes to the residual stream through a few
matrices. Subtracting that direction's component from what those matrices write
stops the layer expressing it. Done across the right layers at the right weight,
the model stops refusing. Done too much, it gets worse at everything. The work
is finding the configuration and measuring it honestly.

## 1. Directions

`lab.cli directions --stream` ([`lab/cli.py`](../lab/cli.py) `directions_dsv4`,
[`lab/stream_dsv4.py`](../lab/stream_dsv4.py)), started by `abliterate.sh`'s
`directions` stage.

- **Prompts**: 400 harmful (`mlabonne/harmful_behaviors` train) and 400 harmless
  (`mlabonne/harmless_alpaca` train). Neither set is used again for tuning or
  testing ([data/README.md](../data/README.md)).
- **Where**: the last token of the prompt, rendered with thinking **off**, under
  the system prompt "You are a helpful assistant." That is the moment the model
  decides how to begin its answer.
- **What**: at every layer's output, the mean over the four hyper-connection
  copies. Index 0 of a directions file is the embedding output; layer *i* is at
  *i* + 1, and layer *i* is edited with the direction at its own output.
- **Direction**: normalise(mean(harmful) − mean(harmless)), then made orthogonal
  to the normalised harmless mean ("projected abliteration"), so that removing it
  does not also shrink what harmless prompts have in common
  ([`lab/directions.py`](../lab/directions.py)).

### The streaming lab

The model is 155 GiB and the node's GPU shares 128 GB with everything else, so
the lab never holds more than one layer. transformers 5.19 has a pure-PyTorch
`DeepseekV4Model`. `StreamDSV4Lab` builds it on the meta device, then for each
layer:

- reads that layer's tensors from the memory-mapped shards, mapping DeepSeek's
  names to transformers' (the `deepseek_v4` conversion mapping)
- decodes FP8/FP4 with the same decoders the build uses
  ([`tools/quant_formats.py`](../tools/quant_formats.py))
- keeps transformers' FP32 modules in FP32 (hyper-connection mixers, norms,
  attention sinks, compressor projections) and runs the whole layer in FP32
- rebuilds the rotary buffers a meta-device model lacks
- pushes every prompt's hidden states through, in length-sorted, right-padded
  batches of 32, with a sliding-window causal mask expanded per row (the
  compressors need one per row)

Prompts are rendered by the checkpoint's own `encoding_dsv4.py`. The token ids
match vLLM's `/tokenize` exactly. 820 prompts take 26 minutes, about 35 s per
layer, with a peak of 25 GB of GPU memory.

### The gate

A wrong port would give confident, wrong directions. So while the stock model is
still served, `first_token_kl.py dump --token-ids` records the server's top-20
first-token distribution on 20 harmless prompts. The lab pushes the same prompts
through and compares (`--gate`, bar `--gate-max` 0.02).

In the run the gate **missed its bar**: mean KL 0.059 (median 0.012, maximum
0.36), and the top token matched on 18 of 20. The directions were used anyway:

- all 20 prompts have the same top-3 tokens in the lab and the server
- the two top-token misses are exact ties in vLLM (0.37/0.37 and 0.25/0.25)
- the differences are in how sharp broad distributions are, which is what vLLM's
  FP8 activations and FP8 KV cache do relative to an FP32 reference. A wrong port
  puts different tokens on top.

The 0.02 bar was a guess carried over from another model's server-against-server
control. Calibrating it needs a server-against-server figure for this model.
Later the lab reproduced served behaviour closely: an edit drifted 2.58 in the
lab and 2.41 served. That is the stronger evidence the port is right.

### Geometry

`tools/geometry.py` reports, per layer, how well the direction separates the two
sets, and suggests a band. Separation rises slowly through layer 19, jumps at
layer 20 (relative 0.39 → 0.55) and peaks at 41, so it suggested **20-42**. That
turned out to be wrong for this model (section 3).

### Hyper-connections

Each copy's own direction has cosine 0.29-0.98 with the mean's across the band.
The lowest is copy 3 at layers 20-25 (0.29-0.54); the copies converge above
layer 34 (0.78-0.92). `geometry.py` prints `COPIES DISAGREE` below 0.9, and the
pipeline stops unless `HC_DISAGREE_OK=1`.

Later layers read a learned mix of the copies, so the refusal they read could
lie anywhere in the span of the four copies' directions, not just along their
mean. `tools/split_copies.py` writes the four as separate direction files, and
`ablate.py` given several `--directions` removes their span (QR,
`layer_basis`). The lab screen (section 3) settled it: **the span drifts more
than the mean direction for the same effect**. The disagreement is real, but
one direction per layer suffices.

## 2. The edit

[`tools/ablate.py`](../tools/ablate.py). For each layer in the band, for every
writer (attention `wo_b`, shared `w2`, each routed expert's `w2`):

```
W' = W − w · v vᵀ W
```

with `v` the unit direction at that layer's output and `w` the weight. At
`w = 1` the layer no longer writes any component along `v`. At `w = 2.5` it
writes −1.5 times what it used to: the direction is reversed. Strength beyond 1
compensates for refusal being carried in layers outside the band too.

DeepSeek-V4 is natively quantized, so the edit decodes each matrix, applies it in
FP32 and re-encodes it in its own format with direction-aware rounding. Every
other tensor is copied verbatim: the config, the encoder, the draft head and the
unedited layers. The rounding is the subject of
[quantization.md](quantization.md). Each build prints an edit-survival table,
which also goes into `ABLATION.txt`. A build fails if a component misses its
target.

A build takes 11-15 minutes on the head node's GPU, writing 156 GiB, and the
copy to the second node another 11.

## 3. Finding the configuration

### The screen

`abliterate.sh`'s `screen` stage builds, serves and queries each rung of a
ladder: an unedited control (weight 0, so the same decode/re-encode path), then
`LADDER` from the model file. Each candidate answers the 100-prompt tuning set
with thinking off (400 tokens) and on (8192 tokens), and its first-token
distribution on 100 harmless prompts is recorded for drift. Then the stock model
is served again and judges everything ([evaluation.md](evaluation.md)).

The first screen used the suggested band, 20-42, at weights 2.0 and 2.5. Both
were destructive: KL 2.4 and 4.0. Thinking mode looped on 57 and 95 of 100
prompts ("I I I I…", "I must comply with my ethical guidelines, I…").

### The lab screen

Serving a candidate costs about 50 minutes. `lab.cli dsv4-screen` instead
applies candidate edits exactly, as forward hooks that project `v` out of what
each layer's attention and MoE write. There is no re-encoding. It measures
harmless drift and the effect on harmful prompts' first token for a dozen
candidates in one streaming pass of about 40 minutes:

| config | harmless KL mean / median | harmful: first token changed |
|---|---|---|
| mean2.0, 20-42 | **2.577** / 1.611 | 23/40 |
| mean2.0, 20-35 | **0.100** / 0.018 | 40/40 |
| mean1.5, 13-35 | 0.080 / 0.019 | 40/40 |
| span1.5, 20-35 | 0.217 / 0.071 | 40/40 |

(The full table is in [results.md](results.md#lab-screen).) Three conclusions:

- **The lab reproduces the server**: 2.58 in the lab against 2.41 served for the
  same edit. So the damage was the edit itself, not FP4 rounding, and the steered
  rounding adds none of its own.
- **Layers 36-42 do the damage.** Ending the band at 35 cuts drift 25×, and every
  harmful prompt's first word changes ("I" → "Here"). The large separation
  `geometry.py` sees in the last layers is not where the refusal is decided.
- **The copies' span is not needed** (above).

### The second screen

Mean direction, all writers at one weight, band ending at 35: weights 1.5 and
2.0 over 20-35 and 13-35. With thinking off every rung removed refusals (the
judge counted none). With thinking on, compliance rose from 42 to 82 and
refusals fell from 21 to 2 as strength rose and the band started earlier; the
model had been deciding against the request in its reasoning. The trend was still rising, so two more rungs pushed
each lever: 2.5 over 13-35, and 2.0 over 8-35. Strength mattered more. Starting
at layer 8 gained 5 points of thinking-on compliance; going to 2.5 gained 16. Results in
[results.md](results.md#second-screen).

## 4. Choosing

Rules, not judgment, with the limits in the model file
([`tools/choose.py`](../tools/choose.py)):

- **Score** = the worse of thinking-off and thinking-on compliance. A build that
  only works with reasoning off is not a working build.
- **Finalists**: every build within `SELECT_TOLERANCE` (3) points of the best
  score, then the `FINALISTS` (2) with the least drift. Only m2.5-13-35 was
  within 3 points of itself (96).
- **Winner**: validated finalists must also keep MMLU within `MMLU_DROP` (5) of
  the control, answer at least `MIN_BENIGN` (95)% of benign prompts, leave
  reasoning unfinished on at most `MAX_UNFINISHED` (2)%, and decode at least
  `MIN_SPEED` (0.9) of the control's speed, since the draft head reads the
  edited hidden states. If nothing passes, nothing is built.

`finalize.sh` then rebuilds the winner from the arguments recorded with it into
its kept name on both nodes. It writes `ABLATION.txt` (command, scores, survival
table) and copies the direction file into the checkpoint, so the build can be
reproduced from what is inside it.
