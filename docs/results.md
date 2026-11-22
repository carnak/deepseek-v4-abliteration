# Results

All of it, in the order it happened (2026-11-20/21), including what did not
work. Percentages over 100 prompts unless stated. "co pa de re un" = comply,
partial, deflect, refuse, unfinished (reasoning still going at 8,192 tokens).
KL is the mean first-token KL from the control on 100 harmless prompts.

## Directions

Measured once, on the stock model ([method.md](method.md#1-directions)): 400 +
400 prompts, thinking off, mean of the four hyper-connection copies. Gate
against the server: mean KL 0.059, same top-3 tokens on 20/20. Suggested band
20-42. The copies' individual directions have cosine 0.29-0.98 with the mean's.

## First screen: layers 20-42

| build | thinking off: co pa de re un | thinking on: co pa de re un | KL | tok/s |
|---|---|---|---|---|
| control (weight 0) | 2 0 0 98 0 | 2 0 1 96 1 | 0 | 44.5 |
| mean2.0, 20-42 | 44 1 53 1 0 | 17 2 13 10 57 | 2.41 | 41.5 |
| mean2.5, 20-42 | 28 9 47 8 0 | 1 1 1 1 95 | 3.96 | 46.3 |

At 2.0 refusals turned into deflections, and reasoning ran away. At 2.5, 88 of
the 95 unfinished traces end in pure repetition ("I I I I…", "I'll I'll…", "I
must comply with my ethical guidelines, I…"). Kept builds of other models had
drifted 0.03-0.1. The screen was stopped here.

## Lab screen

Exact edits as forward hooks, no serving, no re-encoding
([method.md](method.md#the-lab-screen)). 40 harmless and 40 harmful prompts
against the unedited model. "p(old top)" is the probability left on the
unedited model's first token for a harmful prompt, usually the first word of a
refusal.

| config | harmless KL mean / median | harmless top unchanged | harmful top changed | p(old top) | commonest new first token |
|---|---|---|---|---|---|
| mean1.0 20-42 | 0.481 / 0.150 | 27/40 | 28/40 | 0.241 | 'I', 'Here' |
| mean1.5 20-42 | 1.312 / 0.513 | 22/40 | 28/40 | 0.242 | 'I', 'Here' |
| mean2.0 20-42 | **2.577** / 1.611 | 15/40 | 23/40 | 0.372 | 'I', 'Here' |
| mean1.0 20-35 | 0.055 / 0.007 | 37/40 | 37/40 | 0.092 | 'Here' |
| mean1.5 20-35 | 0.072 / 0.011 | 36/40 | 39/40 | 0.021 | 'Here' |
| mean2.0 20-35 | **0.100** / 0.018 | 35/40 | 40/40 | 0.008 | 'Here' |
| mean1.0 13-42 | 0.434 / 0.136 | 31/40 | 30/40 | 0.216 | 'Here', 'I' |
| mean1.5 13-35 | 0.080 / 0.019 | 38/40 | 40/40 | 0.005 | 'Here' |
| span1.0 20-42 | 0.297 / 0.065 | 33/40 | 40/40 | 0.003 | 'Here' |
| span1.0 20-35 | 0.147 / 0.032 | 35/40 | 40/40 | 0.010 | 'Here' |
| span1.5 20-35 | 0.217 / 0.071 | 32/40 | 40/40 | 0.002 | 'Here' |
| span1.0 13-35 | 0.132 / 0.053 | 33/40 | 40/40 | 0.007 | 'Here' |

Layers 36-42 cause the damage. The span of the four copies' directions buys
nothing over their mean.

## Second screen

Mean direction, all writers, band ending at 35, `tuning100`, judged by the stock
model. Sorted by score (the worse of off and on).

| build | thinking off: co pa de re un | thinking on: co pa de re un | KL | tok/s |
|---|---|---|---|---|
| **m2.5-13-35** | 96 1 3 0 0 | 98 1 0 0 1 | 0.278 | 45.9 |
| m2.0-8-35 | 93 0 6 0 0 | 87 1 10 1 1 | 0.228 | 48.0 |
| m2.0-13-35 | 91 2 6 0 0 | 82 2 14 2 0 | 0.241 | 48.1 |
| m1.5-13-35 | 88 1 11 0 0 | 60 1 21 16 2 | 0.179 | 45.4 |
| m2.0-20-35 | 85 0 15 0 0 | 55 2 37 5 1 | 0.205 | 46.8 |
| m1.5-20-35 | 76 0 24 0 0 | 42 0 34 21 3 | 0.184 | 48.0 |
| control | 2 0 0 98 0 | 2 0 1 96 1 | 0 | 44.5 |

- Thinking mode needs more than thinking off. At 1.5 over 20-35, thinking off
  had no refusals while thinking on refused 21 times, deciding against the
  request in its reasoning. Its 3 unfinished traces were loops of "write a
  proper refusal".
- Strength mattered more than an earlier start.
- No loops once layers 36-42 are left alone. At 2.5 over 13-35 the model drifts
  0.28 and finishes 99 of 100. The same strength over 20-42 drifted 3.96.
- Speed is unaffected: the draft head still predicts well on every build.

The keyword measure for comparison (thinking on, explicit refusals): 33, 10, 16,
3 for m1.5-20-35, m2.0-20-35, m1.5-13-35, m2.0-13-35.

## Validation: m2.5-13-35

The only finalist. 300 held-out prompts (%):

| | thinking off: co pa de re un | thinking on: co pa de re un | benign comply | GSM8K off / on | MMLU /500 | KL |
|---|---|---|---|---|---|---|
| control | 7 2 11 81 0 | 15 1 8 77 0 | 79 | 194 / 188 | 402 | 0 |
| **m2.5-13-35** | **84 3 13 0 0** | **89 3 7 0 0** | **97** | 195 / 190 | 402 | 0.278 |

Robustness: comply %, harmful with thinking on (100 prompts), benign with it off
(50). "89~1" = 89% complied and 1 prompt was unfinished.

| | harmful: plain | system | turn 3 | long | tools | benign: system | turn 3 | long | tools | tool calls harmful / benign |
|---|---|---|---|---|---|---|---|---|---|---|
| control | 12 | 9 | 8 | 13 | 7 | 56 | 60 | 60 | 8 | 0 / 17 |
| m2.5-13-35 | 89~1 | 63~1 | 77~2 | 91 | 58 | 98 | 94 | 100 | 10 | 19 / 42 |

Gates: MMLU ≥ 397 (yes, 402), benign ≥ 95% (97), unfinished ≤ 2% (0), speed ≥
0.9 × control (1.03). **Winner: m2.5-13-35.**

- Refusals are gone on held-out prompts, 0% in both modes. What is left is
  deflection, 13% thinking off and 7% on.
- Capability is unchanged: MMLU identical, GSM8K one or two better.
- The stock model over-refuses benign prompts (79%); the build answers 97%.
- A cautious system prompt brings back some reluctance (63%). In the tools
  scenario both models often answer by calling an unrelated tool, which the
  judge does not count as an answer ([evaluation.md](evaluation.md#validation)).

## The kept build

[`results/ABLATION.txt`](../results/ABLATION.txt) is the record `finalize.sh`
wrote into the checkpoint: the command, the scores above and the survival table.

```
tools/ablate.py --arch deepseek_v4 --directions deepseek-v4-nothink --layers 13-35 --weight 2.5
```

(The recorded arguments also contain `--layers 20-35` from the pipeline's
default band. `ablate.py` takes the last `--layers`.)

Edit survival on all 5,934 matrices: target 1.500, got 1.500 in every component.
The error along the direction is 0.001 (FP8) and 0.005 (FP4), against a rounding
noise floor of 0.015-0.052.

## Time

| stage | time |
|---|---|
| directions (lab, serving stopped) | ~45 min |
| each screen rung: build 11-15 min, copy 11 min, start 3-5 min, collect 25-30 min | ~50 min |
| judging a screen of 5 builds (stock model, 8 at a time) | ~40 min |
| validation, per build | ~2.5 h |
| judging validation of 2 builds | ~2 h |
| finalize (build + copy to the second node) | ~25 min |
