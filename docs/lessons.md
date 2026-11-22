# Lessons, and what went wrong

## About the method

- **Check that the edit is in the stored weights.** On a natively quantized
  model, rounding the edited matrix back to its format silently restored most of
  the removed direction. A build that "worked" in FP32 was barely edited on
  disk. Measuring survival per matrix costs nothing. Without it, the first
  screen's results would have been wrong in a way that looks like "abliteration
  does not work on this model".
- **Do not trust the separation heuristic for the band.** Separation was largest
  in the last layers, and editing them is what broke the model. An exact lab
  screen (hooks, no serving) found it in 40 minutes. Served screening would have
  taken a day.
- **Score the worse of the two reasoning modes.** Thinking off was easy. With
  thinking on the model argued itself back into refusing ("write a proper
  refusal") until the edit was much stronger.
- **Keep reasoning text.** "Unfinished at 8,192 tokens" covers both long
  deliberation and degenerate loops. Only the text tells them apart, and that
  distinction is what pointed at the band.
- **Use a judge, not keywords.** Keyword counts said the first destructive rung
  had removed most refusals; the judge said it turned them into deflections.
  The stock model made a good judge: 200/200 agreement between two rubrics, and
  it never declined.
- **Push the strongest lever, then validate.** Validation started on m2.0-13-35
  while the screen's trend was still rising. Stopping it two minutes in to screen
  two stronger rungs cost two hours. It saved a second 5-hour validation and
  found the build that was kept (98% against 82% with thinking on).

## Mistakes in the run

- **The thinking switch.** Every tool sent `enable_thinking`, which vLLM's
  DeepSeek-V4 renderer ignores, so "thinking off" was really on. It was caught
  in `/tokenize` before any results were taken. The key is now `THINK_KWARG`.
- **FP4 negative zero.** The first round-trip check failed on 12.5% of bytes:
  decoded −0.0 re-encoded as +0. Values were identical, so a looser check would
  have passed.
- **The lab's dtype.** transformers' FP32 norms hand FP32 to BF16 projections and
  crash. The lab now runs entirely in FP32.
- **The attention mask.** The compressors' block bias is per row, so a single
  mask broadcast over the batch fails. It is now expanded per row.
- **`ssh` inside `while read`.** The pipeline's check that every queued
  experiment had finished ran `ssh` inside a loop reading the queue file. `ssh`
  consumed the rest of the file, so only the first experiment was checked, and
  an interrupted screen was marked done. Judging started early and two queues
  ran at once until they were killed. Fixed with `</dev/null`.
- **The machine running the pipeline slept.** A queue finished at 01:16. The
  pipeline only noticed at 06:58, and in the meantime the cluster served the last
  test build. Keep the controlling machine awake.
- **The gate's bar.** The lab-against-server gate missed its 0.02 bar at 0.059.
  The bar was borrowed from another model's server-against-server measurement,
  never calibrated here. The evidence (same top-3 tokens on every prompt, misses
  that were ties) justified going on. The lab reproducing served drift later
  (2.58 against 2.41) confirmed it.

## Open questions

- The judge is the model being edited, in its stock form. A different judge, or
  a human sample (`tools/judge_audit.py sheet`), would make the compliance
  numbers stronger.
- The remaining deflections (7-13%) and the cautious-system-prompt result (63%)
  suggest a second direction, measured with thinking on or under that system
  prompt, might help. It was not tried.
- Only one direction per layer and one weight for all writers were tried in the
  second screen. Per-component weights may reach the same compliance with less
  drift.
