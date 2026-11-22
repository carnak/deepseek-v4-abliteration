# DeepSeek-V4-Flash-0731: what abliteration has to know

Read from the checkpoint's `config.json`, its safetensors headers and DeepSeek's
reference `inference/model.py`.

`model_type` is `deepseek_v4`: 43 layers, hidden size 4096, 284B parameters with
13B active per token. Every layer has the same structure:

- **Attention**: MLA-style low-rank queries, one shared KV head, a sliding
  window of 128, and per layer either Compressed Sparse Attention (CSA:
  compresses every 4 tokens and picks the top 512 with an indexer) or Heavily
  Compressed Attention (HCA: every 128 tokens).
- **MoE**: 256 routed experts (top 6) plus one shared expert. The first 3 layers
  are hash-routed: tokens go to experts by a fixed table.
- A **DSpark draft head** (`mtp.*`) predicts several tokens ahead for
  speculative decoding. It reads the main model's hidden states.

## Hyper-connections

The residual stream is not one vector per token but four (`hc_mult` 4):
`[batch, seq, 4, 4096]`.

- A sublayer reads a learned, input-dependent mix of the four copies (`hc_pre`).
- Its single 4096-wide output is added to all four copies, each scaled by its
  own weight (`hc_post`).
- The copies are then mixed by a doubly-stochastic matrix (Sinkhorn-normalised).

Two consequences for abliteration:

- **The edit still works.** Every writer writes one vector, added to each copy
  with some scale. Removing `v` from what a writer writes (`W' = W − w·v vᵀW`)
  removes it from what that writer adds to every copy.
- **Where to measure is a choice.** Each copy has its own activations, so there
  are four candidate directions per layer, plus their mean. The pipeline uses
  the mean of the four copies, which is also the reduction the draft head reads
  (`h.mean(dim=2)`). It saves each copy's direction too, and
  `tools/geometry.py` compares them (see [method.md](method.md#hyper-connections)).

## The residual-stream writers

These are the matrices abliteration edits, all `[out = 4096, in]`:

| component | tensor | shape | stored as |
|---|---|---|---|
| attn | `layers.<i>.attn.wo_b.weight` | [4096, 8192] | FP8 E4M3, UE8M0 scale per 128×128 block |
| shared | `layers.<i>.ffn.shared_experts.w2.weight` | [4096, 2048] | FP8 E4M3, 128×128 blocks |
| experts | `layers.<i>.ffn.experts.<e>.w2.weight` ×256 | [4096, 2048] | FP4 E2M1 packed two per int8, UE8M0 scale per 32 inputs |

- `wo_b` is the second half of the grouped output projection: `wo_a` maps each
  of the 8 head groups to 1024 dimensions, and `wo_b` mixes them to the hidden
  size. It is attention's only writer.
- Each `.weight` has a `.scale` tensor beside it. UE8M0 is an unsigned 8-bit
  exponent: every scale is a power of two.
- The draft head's weights are not edited. It does not write the main model's
  residual stream, and its acceptance rate is measured instead, as decode speed.

`tools/arch.py` (`class DeepseekV4`) names these per layer. Editing all three
components over layers 13-35 is 23 × (1 + 1 + 256) = 5,934 matrices.

## Prompt format and the thinking switch

DeepSeek-V4 has no Jinja chat template. Prompts are rendered by the checkpoint's
`encoding/encoding_dsv4.py`. In chat mode (thinking off) a single-turn prompt is

```
<bos>{system}<｜User｜>{prompt}<｜Assistant｜></think>
```

and in thinking mode the assistant turn opens `<think>` instead. vLLM's
DeepSeek-V4 renderer switches between them with the chat-template kwarg
**`thinking`**. It silently ignores keys it does not know, so the common
`enable_thinking: false` leaves reasoning on. The tools read the key's name
from `THINK_KWARG` (`tools/thinking.py`). This was checked with `/tokenize`
(token 128821 `<think>` against 128822 `</think>`) and in responses: the wrong
key gave 206-458 characters of reasoning on plain questions, the right one gave
none.

The served `reasoning_effort` was `medium`. The model card names low, high and
max; the server accepts medium and it behaves as low.
