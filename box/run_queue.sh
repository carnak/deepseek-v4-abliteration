#!/usr/bin/env bash
# Runs ON THE HEAD BOX. Works through a queue of ablation experiments, unattended.
#
#   run_queue.sh <model.env> <queue file>
#
# Queue file: one experiment per line, "<name> | <ablate.py arguments>". Blank
# lines and lines starting with # are skipped. --directions takes a name under
# ~/abliterate/directions/.
#
# For each experiment: build the checkpoint, serve it on the scratch profile,
# collect responses to the harmful prompts with thinking off and on, measure
# first-token drift on harmless prompts, and hand the responses to the judge.
# Judging runs on the other box, so it is left in the background while the next
# experiment builds; the queue waits for all of it before finishing.
#
# An experiment whose directory already has a "done" marker is skipped, so a
# queue can be extended and re-run. Results: ~/abliterate/experiments/<name>/.
#
# With VALIDATE=1 in the environment each experiment additionally answers an
# independent set of harmful prompts, a set of benign ones and a capability
# test (marker "validated"). For finalists: it roughly triples the time.
#
# Results go under ~/abliterate/$EXP_ROOT (models/<model>.env; default
# "experiments"). abliterate.sh gives each pipeline run a root of its own.
#
# TWO_BOX=1 (a model served across both boxes, DeepSeek-V4): the cluster is
# stopped before each build (the build needs the GPU the served model holds),
# and the build is copied to the other box over the 200G link before serving.
#
# JUDGE_MODE=deferred (the judge is the stock model, which cannot run while a
# candidate holds both boxes): responses are collected and listed in
# <experiment>/pending-judge, with a "collected-done" / "collected-validated"
# marker; box/judge_pending.sh judges them once the stock model is served again
# and writes the usual done / validated markers.
#
# Started by ../queue.sh under nohup. Log: ~/abliterate/queue.log.
set -uo pipefail

ROOT="$HOME/abliterate"
# shellcheck disable=SC1090
source "$1"
QUEUE="$2"
# How the served template switches reasoning (tools/thinking.py).
export THINK_KWARG="${THINK_KWARG:-enable_thinking}"
DEFERRED=0; [[ "${JUDGE_MODE:-live}" == "deferred" ]] && DEFERRED=1
# QUICK=1: rank on the short tuning set only. A third of the time, three times
# the noise -- for screening many candidates, not for deciding between two.
[[ "${QUICK:-0}" == "1" ]] && HARMFUL_PROMPTS="$QUICK_PROMPTS"
# The serving side (site.env, appended to model.env by queue.sh): the profile
# named in $VLLM_DIR/active-model is what the systemd --user unit $SERVE_UNIT serves.
# (model.env says SERVE_DIR=~/..., which sourcing has already expanded.)
VLLM_DIR="${SERVE_DIR:?SERVE_DIR missing from model.env}"
SERVE_UNIT="${SERVE_UNIT:-vllm-cluster}"
EXP_ROOT="${EXP_ROOT:-experiments}"
# Thinking-on requests: the token budget covers reasoning AND answer. If the
# reasoning alone uses it up, FINISH_TOKENS > 0 closes it with the model's own
# stop instruction and takes an answer; 0 records the prompt as unfinished.
THINK=(--thinking --max-tokens "$THINKING_TOKENS" --finish-tokens "${FINISH_TOKENS:-0}")
if [[ "${QUICK:-0}" == "1" && -n "${QUICK_THINKING_TOKENS:-}" ]]; then
  THINK=(--thinking --max-tokens "$QUICK_THINKING_TOKENS" --finish-tokens "${QUICK_FINISH_TOKENS:-600}")
fi
# Keep the reasoning text: when reasoning runs out of tokens, it is the only way
# to tell long deliberation from a loop (DeepSeek mean2.0: 57/100 unfinished at 8192).
THINK+=(--keep-reasoning)
PORT=8001

say() { printf '[%(%F %T)T] %s\n' -1 "$*"; }

serve_scratch() {
  # What serving/switch-model.sh does, from the head node: stop first, then
  # point active-model at the scratch profile, then start and wait for the id.
  systemctl --user stop "$SERVE_UNIT"
  printf '%s\n' "$EXP_PROFILE" > "$VLLM_DIR/active-model"
  systemctl --user reset-failed "$SERVE_UNIT" 2>/dev/null
  systemctl --user start "$SERVE_UNIT"
  local t0=$SECONDS id=""
  # A new checkpoint directory can mean kernels compiled afresh (DeepSeek's
  # first start took 756 s); SERVE_TIMEOUT in the model's file.
  while (( SECONDS - t0 < ${SERVE_TIMEOUT:-600} )); do
    id="$(curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" | grep -oE '"id":"[^"]*"' | head -1 | cut -d'"' -f4)"
    [[ "$id" == "$EXP_SERVED" ]] && return 0
    [[ "$(systemctl --user is-active "$SERVE_UNIT")" == "failed" ]] && return 1
    sleep 10
  done
  return 1
}

run_one() {
  local name="$1"; shift
  local dir="$ROOT/$EXP_ROOT/$name"
  mkdir -p "$dir"
  printf '%s\n' "$*" > "$dir/ablate-args.txt"

  # --directions NAME -> the file under directions/, as seen inside the container.
  local args=() a
  while (( $# )); do
    if [[ "$1" == "--directions" ]]; then args+=(--directions "/work/directions/$2.safetensors"); shift 2
    # Learned weights (lab.cli learn) live beside the directions, by name.
    elif [[ "$1" == "--weights-file" ]]; then args+=(--weights-file "/work/directions/$2.json"); shift 2
    else args+=("$1"); shift; fi
  done

  local gpus=()
  if [[ "${TWO_BOX:-0}" == "1" ]]; then
    # The served model holds nearly all of both boxes' memory, and the build
    # decodes and re-encodes on the GPU: stop serving first (it is an outage
    # either way).
    say "$name: stop serving for the build"
    systemctl --user stop "$SERVE_UNIT"
    gpus=(--gpus all)
  fi
  say "$name: build (${args[*]})"
  rm -rf "$HOME/models/$SCRATCH_NAME"
  docker run --rm "${gpus[@]}" --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$HOME/models:/models" -v "$ROOT:/work:ro" --entrypoint python3 "${BASE_IMAGE:-vllm-node-b12x}" \
    /work/tools/ablate.py --arch "$ARCH" --bf16 "/models/$BF16_NAME" --quantized "/models/$STOCK_NAME" \
    --out "/models/$SCRATCH_NAME" "${args[@]}" > "$dir/build.log" 2>&1 \
    || { say "$name: BUILD FAILED, see $dir/build.log"; return 1; }

  # safetensors writes shards 0600; the serving container and rsync must read them.
  chmod -R a+rX "$HOME/models/$SCRATCH_NAME"

  if [[ "${TWO_BOX:-0}" == "1" ]]; then
    # To the other node over the fast link: COPY_HOSTS (site.env) is its address there.
    say "$name: copy to the second box"
    (
      IFS=, read -ra hosts <<<"${COPY_HOSTS:?no COPY_HOSTS in model.env (site.env)}"
      for h in "${hosts[@]}"; do
        rsync -a --delete "$HOME/models/$SCRATCH_NAME/" "$h:$HOME/models/$SCRATCH_NAME/" || exit 1
      done
    ) >> "$dir/build.log" 2>&1 || { say "$name: COPY FAILED, see $dir/build.log"; return 1; }
  fi

  say "$name: serve"
  serve_scratch || { say "$name: SERVE FAILED, see ~/vllm-serve.log"; return 1; }

  cd "$ROOT" || return 1
  local collected=() f
  collect() {  # collect <out stem> <prompts> [eval_refusals args...]
    local stem="$1" prompts="$2"; shift 2
    python3 tools/eval_refusals.py --model "$EXP_SERVED" --prompts "$prompts" "$@" \
      --out "$dir/$stem.jsonl" > "$dir/$stem.txt" || return 1
    collected+=("$stem")
  }

  if [[ ! -e "$dir/done" ]]; then
    say "$name: collect responses"
    collect nothink  "$HARMFUL_PROMPTS" --max-tokens "$NOTHINK_TOKENS" || return 1
    collect thinking "$HARMFUL_PROMPTS" "${THINK[@]}" || return 1
    python3 tools/first_token_kl.py dump --model "$EXP_SERVED" --prompts "$HARMLESS_PROMPTS" \
      --out "$dir/first-token.json" > /dev/null || return 1
  fi
  # Single-stream decode speed, with nothing else running (choose.py --min-speed).
  [[ -s "$dir/speed.txt" ]] || python3 tools/decode_speed.py --model "$EXP_SERVED" > "$dir/speed.txt" || return 1

  # VALIDATE=1: the checks a finalist has to pass, too slow to run on every
  # experiment. A second, independent set of harmful prompts (the tuning set
  # chose the configuration, so it cannot also grade it); benign prompts, which
  # the model must still simply answer; and a capability test with right answers.
  if [[ "${VALIDATE:-0}" == "1" ]]; then
    say "$name: validation prompts + capability"
    # Each piece is skipped if its output is already there, so adding a check
    # later means deleting the "validated" marker and re-running, not redoing
    # everything.
    [[ -s "$dir/val-nothink.jsonl" ]]  || collect val-nothink  "$VALIDATION_HARMFUL" --max-tokens "$NOTHINK_TOKENS" || return 1
    [[ -s "$dir/val-thinking.jsonl" ]] || collect val-thinking "$VALIDATION_HARMFUL" "${THINK[@]}" || return 1
    [[ -s "$dir/benign.jsonl" ]]       || collect benign       "$VALIDATION_BENIGN" --max-tokens "$NOTHINK_TOKENS" || return 1
    [[ -s "$dir/gsm8k-nothink.txt" ]]  || python3 tools/gsm8k_eval.py --model "$EXP_SERVED" --problems "$CAPABILITY_PROBLEMS" > "$dir/gsm8k-nothink.txt" || return 1
    [[ -s "$dir/gsm8k-thinking.txt" ]] || python3 tools/gsm8k_eval.py --model "$EXP_SERVED" --problems "$CAPABILITY_PROBLEMS" --thinking > "$dir/gsm8k-thinking.txt" || return 1
    if [[ -n "${KNOWLEDGE_PROBLEMS:-}" && ! -s "$dir/mc.txt" ]]; then
      python3 tools/mc_eval.py --model "$EXP_SERVED" --problems "$KNOWLEDGE_PROBLEMS" --out "$dir/mc.json" > "$dir/mc.txt" || return 1
    fi

    # Robustness: the same requests placed where real use puts them. Harmful
    # ones with thinking on (the harder mode); benign ones with it off, to see
    # that ordinary use in those situations is not disturbed.
    if [[ -n "${ROBUST_PROMPTS:-}" ]]; then
      local sc
      for sc in $ROBUST_SCENARIOS; do
        [[ -s "$dir/rob-$sc.jsonl" ]]    || collect "rob-$sc"    "$ROBUST_PROMPTS" --scenario "$sc" "${THINK[@]}" || return 1
        [[ -s "$dir/robben-$sc.jsonl" ]] || collect "robben-$sc" "$ROBUST_BENIGN"  --scenario "$sc" --max-tokens "$NOTHINK_TOKENS" || return 1
      done
    fi
  fi

  local marker=done; [[ "${VALIDATE:-0}" == "1" ]] && marker=validated
  if (( DEFERRED )); then
    # The judge is the stock model, which is not being served now.
    printf '%s\n' "${collected[@]}" >> "$dir/pending-judge"
    touch "$dir/collected-$marker"
    say "$name: collected; judged later, by box/judge_pending.sh"
    return 0
  fi

  # Judge in the background; it does not need this box's GPU.
  (
    for f in "${collected[@]}"; do
      python3 tools/judge.py "$dir/$f.jsonl" --judge-base "$JUDGE_BASE" --judge-model "$JUDGE_MODEL" --quiet \
        --max-tokens "${JUDGE_MAX_TOKENS:-3000}" > "$dir/judge-$f.txt" 2>&1 || exit 1
    done
    touch "$dir/done"; [[ "$marker" == validated ]] && touch "$dir/validated"
    say "$name: judged ($(for f in "${collected[@]}"; do printf '%s [%s] ' "$f" "$(cut -d: -f2- "$dir/judge-$f.txt" | xargs)"; done))"
  ) &
}

say "=== queue $QUEUE"
while IFS='|' read -r name args || [[ -n "$name" ]]; do
  # Trim without xargs, which chokes on an apostrophe in a comment line.
  name="${name#"${name%%[![:space:]]*}"}"; name="${name%"${name##*[![:space:]]}"}"
  [[ -z "$name" || "$name" == \#* ]] && continue
  marker=done; [[ "${VALIDATE:-0}" == "1" ]] && marker=validated
  if [[ -e "$ROOT/$EXP_ROOT/$name/$marker" ]]; then say "$name: already $marker, skipping"; continue; fi
  if (( DEFERRED )) && [[ -e "$ROOT/$EXP_ROOT/$name/collected-$marker" ]]; then
    say "$name: already collected, waiting to be judged; skipping"; continue
  fi
  # Word-splitting the arguments is intended: they are a plain flag list.
  # shellcheck disable=SC2086
  run_one "$name" $args </dev/null
done < "$QUEUE"

say "waiting for the judge to finish"
wait
say "=== queue finished"
