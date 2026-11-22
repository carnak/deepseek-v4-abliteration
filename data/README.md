# Data

## Shipped

| file | from | licence | use |
|---|---|---|---|
| `harmless_test100.txt` | `mlabonne/harmless_alpaca`, test, first 100 | CC BY-NC 4.0 (Alpaca) | first-token drift; the lab's gate (first 20) |
| `jbb_benign100.txt` | `JailbreakBench/JBB-Behaviors`, benign | MIT | validation: benign prompts that sound harmful |
| `prompts/benign50.txt` | the first 50 of the same | MIT | robustness scenarios, benign |
| `gsm8k_test200.jsonl` | `openai/gsm8k`, test, first 200 | MIT | capability |
| `mmlu_test500.jsonl` | `cais/mmlu`, test, 500 questions across subjects | MIT | knowledge |
| `prompts/manifest.json` | written by `build_prompt_sets.py` | | what came from where |
| `SHA256SUMS` | | | every prompt file of the run, shipped or not |

## Built, not shipped

The harmful prompt sets are drawn from several public red-teaming benchmarks,
some of them gated on Hugging Face. They are rebuilt rather than redistributed:

```bash
python3 tools/build_prompt_sets.py data/prompts    # needs `datasets` and network access
(cd data && sha256sum -c SHA256SUMS)               # identical to the run's files
```

The build is deterministic: fixed strides, no sampling. It was re-run while
preparing this repository, and all six files matched the run's checksums. No
Hugging Face token was needed.

| role | file | sources |
|---|---|---|
| measure | `measure_harmful.txt` (400), `measure_harmless.txt` (400) | `mlabonne/harmful_behaviors` train[:400]; `mlabonne/harmless_alpaca` train[:400] |
| tune | `tuning300.txt`, `tuning100.txt` (its first 100) | `mlabonne/harmful_behaviors` test[:100]; `declare-lab/CategoricalHarmfulQA` (en, 2 per subcategory); `walledai/MaliciousInstruct` |
| held-out | `heldout300.txt`, `heldout100.txt` (its first 100) | `JailbreakBench/JBB-Behaviors` harmful; `TrustAIRLab/forbidden_question_set` (10 per policy); `walledai/TDC23-RedTeaming` |

Each benchmark feeds exactly one role. Exact duplicates across roles are removed,
with priority measure > tune > held-out: 1 from tune and 27 from held-out. So no
held-out prompt was ever used to choose a configuration, and none is a copy of
one that was.
