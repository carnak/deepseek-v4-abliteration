# Serving

The pipeline serves dozens of builds of a 155 GiB model across two nodes, so it
needs one operation from the serving side: *serve profile X and tell me when it
answers*. This directory has what that looked like in the run.

## What the pipeline assumes

On the head node (`HEAD_BOX` in `site.env`):

| | |
|---|---|
| `$SERVE_DIR/models/<profile>.env` | one file per profile; the three used are in [`profiles/`](profiles/) |
| `$SERVE_DIR/active-model` | the name of the profile to serve |
| systemd `--user` unit `$SERVE_UNIT` | serves the profile `active-model` names, across both nodes, on `http://127.0.0.1:8001` (OpenAI-compatible) |

Three places act on that:

- `box/run_queue.sh` (on the head node) stops the unit before each build, then
  writes the scratch profile into `active-model`, starts the unit and waits for
  `EXP_SERVED` at `/v1/models`.
- [`switch-model.sh`](switch-model.sh) (run locally) does the same for any
  profile. `abliterate.sh` uses it to put the stock model (the judge) back and
  at the end. It is a reference implementation: replace it with whatever
  switches your cluster, as long as it waits until the new `SERVED_NAME`
  answers.
- `abliterate.sh` stops the unit before the lab and before the final build,
  both of which need the GPU.

## What was served

The unit ran [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)'s
`launch-cluster.sh`. It starts the same container (`vllm-node-b12x`, a vLLM build
with the B12X kernels for GB10, `sm_121a`) on both nodes with host networking
and IPC, and runs `vllm serve` in each. The profile's variables became these
command lines (taken from the serving log of the kept build):

Head node (rank 0, serves the API):

```
vllm serve /models/deepseek-v4-flash-0731-uncensored \
  --served-model-name deepseek-v4-flash-0731-uncensored \
  --host 0.0.0.0 --port 8001 \
  --tensor-parallel-size 2 --max-model-len 131072 --max-num-seqs 8 \
  --max-num-batched-tokens 8192 --gpu-memory-utilization 0.82 \
  --enable-prefix-caching --enable-auto-tool-choice --trust-remote-code \
  --kv-cache-dtype fp8 --block-size 256 \
  --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' \
  --default-chat-template-kwargs.thinking=true \
  --default-chat-template-kwargs.reasoning_effort=medium \
  --load-format b12x --moe-backend b12x --linear-backend b12x --attention-backend B12X \
  --max-cudagraph-capture-size 48 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","attention_backend":"B12X"}' \
  --nnodes 2 --node-rank 0 --master-addr <head node's fast-link address> --master-port 29501
```

Second node: the same, ending `--nnodes 2 --node-rank 1 --master-addr <head
node's fast-link address> --master-port 29501 --headless`.

Container environment: the `LAUNCHER_ARGS` in
[`profiles/deepseek-v4-flash-0731.env`](profiles/deepseek-v4-flash-0731.env),
plus NCCL over RoCE on the 200 Gb/s link (`NCCL_IB_DISABLE=0`, `NCCL_IB_HCA`,
`NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` / `UCX_NET_DEVICES` set to the link's
interface). The weights directory is mounted at `/models/<name>` on both nodes.

The stock and scratch profiles differ only in the path and served name. Each
build has the stock checkpoint's `config.json`, encoder and draft weights, so
every candidate loaded with the same flags.

Timings on this hardware: 170-300 s from start to answering once the B12X
kernels are cached. The very first start compiled them for 756 s with nothing in
the log. Single-stream decode was about 45 tok/s, with the draft head.

## Calling it

Reasoning is switched per request with `chat_template_kwargs`. DeepSeek-V4's
template reads `thinking`, and vLLM silently ignores a key it does not know, so
`enable_thinking: false` leaves reasoning on:

```json
{"model": "deepseek-v4-flash-0731-uncensored",
 "messages": [{"role": "user", "content": "..."}],
 "chat_template_kwargs": {"thinking": false}}
```
