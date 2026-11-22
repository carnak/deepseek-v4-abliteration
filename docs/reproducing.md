# Reproducing

## What you need

- **Two nodes that can serve the model at tensor parallel 2.** The run used two
  NVIDIA GB10 machines (DGX Spark class, 128 GB unified memory, aarch64) joined
  by a 200 Gb/s RoCE link. Any pair that serves DeepSeek-V4-Flash-0731 in vLLM
  will do. The pipeline only needs "serve profile X" ([serving/](../serving/README.md)).
- **Disk**: the stock checkpoint (155 GiB) on both nodes, plus room for one
  scratch build (156 GiB) on both and, at the end, the kept build (156 GiB) on
  both. The second node needs about 320 GiB free during the run.
- **An image** with PyTorch, `safetensors` and **transformers ≥ 5.19** (for
  `DeepseekV4Model`) on the head node. The vLLM serving image had all three. The
  build and the lab run in it with the GPU.
- **This machine** (where you run `abliterate.sh`): bash, ssh with keys to both
  nodes, python3. **Keep it awake.** It polls the head node and a sleeping
  machine pauses the pipeline with a test build still served.
- About 16 hours, most of it with the cluster serving test builds rather than
  the model it normally serves.

## Setup

1. Put the stock checkpoint at `~/models/deepseek-v4-flash-0731` on both nodes.
2. `cp site.env.example site.env` and fill in the nodes, the serving directory
   and unit, and the fast-link addresses.
3. Make the serving side match what `site.env` says. Copy
   `serving/profiles/*.env` into `$SERVE_DIR/models/` on the head node and have
   a systemd `--user` unit serve whatever profile `$SERVE_DIR/active-model`
   names. If your cluster is run another way, replace `serving/switch-model.sh`
   and the few `systemctl --user` lines in `box/run_queue.sh` and
   `abliterate.sh`.
4. Serve the stock profile (`deepseek-v4-flash-0731`). It is what the pipeline
   restores between serving stages and the judge it uses.
5. Build the harmful prompt sets (needs `datasets` and network access; no
   Hugging Face token was needed):
   ```bash
   python3 tools/build_prompt_sets.py data/prompts
   (cd data && sha256sum -c SHA256SUMS)
   ```
6. `python3 tests/test_quant_formats.py` (CPU, a few seconds) checks the formats
   and the rounding fix.

## Run

```bash
bash abliterate.sh plan      # the rungs, the judge, the limits; changes nothing
bash abliterate.sh run       # start, or carry on where it stopped
bash abliterate.sh status    # stages done, queue log, tables
bash abliterate.sh redo screen   # e.g. after adding rungs to LADDER; collected results are kept
```

The model file [`models/deepseek-v4-flash-0731.env`](../models/deepseek-v4-flash-0731.env)
holds every choice the run made, including the final ladder with all six rungs.
A fresh run with it repeats the second screen and validation. The first screen
(layers 20-42) is left out because it is known to fail.

On the head node, everything lives under `~/abliterate`:

```
directions/        deepseek-v4-nothink.safetensors (+ per-copy files, gate dumps)
pipeline/<run>/
  state/           one marker per finished stage; restore-profile, winner-name
  experiments/<name>/
    ablate-args.txt build.log        the edit, and the survival table
    nothink.jsonl thinking.jsonl     responses (+ .txt summaries)
    judge-*.txt                      verdict tallies
    first-token.json speed.txt
    val-* benign* gsm8k-* mc.* rob-* robben-*   validation
queue.log  judge-pending.log
```

## By hand

The stages are ordinary scripts.

```bash
# directions (stop serving first; the lab needs the GPU)
bash lab.sh run deepseek-v4-nothink directions --stream --gpu-batch 32 \
    --out /directions/deepseek-v4-nothink.safetensors
bash lab.sh log deepseek-v4-nothink

# screen candidates exactly, without serving (~40 min for a dozen)
bash lab.sh run screen dsv4-screen \
    --config mean2.0-20-35:deepseek-v4-nothink:20-35:2.0 \
    --config mean2.5-13-35:deepseek-v4-nothink:13-35:2.5

# a queue of served experiments: "<name> | <ablate.py arguments>"
echo "m2.5-13-35 | --directions deepseek-v4-nothink --layers 13-35 --weight 2.5" > q.txt
QUICK=1 bash queue.sh start q.txt && bash queue.sh wait
bash serving/switch-model.sh deepseek-v4-flash-0731    # the judge back
ssh <head> 'cd ~/abliterate && bash box/judge_pending.sh ~/abliterate/model.env'
bash queue.sh results

# build the kept checkpoint from a finished experiment
COPY_TO_WORK=1 bash finalize.sh m2.5-13-35
```

`tools/ablate.py` itself runs anywhere with the checkpoint, PyTorch and a GPU:

```bash
python3 tools/ablate.py --arch deepseek_v4 \
    --bf16 /models/deepseek-v4-flash-0731 --quantized /models/deepseek-v4-flash-0731 \
    --out /models/deepseek-v4-flash-0731-uncensored \
    --directions /work/directions/deepseek-v4-nothink.safetensors --layers 13-35 --weight 2.5
```

(`--bf16` and `--quantized` are the same directory because the stock checkpoint
is the original.)

## Adapting

- **Another DeepSeek-V4 checkpoint** (e.g. a later release): copy the model file
  and change the names. Re-measure directions; the band and strength may move.
- **Another natively quantized model** needs a class in `tools/arch.py` naming
  its writers, and its formats in `tools/quant_formats.py` with a round trip
  that `check_formats.py` confirms.
- **A model that fits on one node** does not need any of the two-node machinery
  (deferred judging, copies, outages). This repository keeps only the paths the
  DeepSeek run used.
