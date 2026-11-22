#!/usr/bin/env bash
# One command: from the stock weights to a validated, named, abliterated build.
#
#   bash abliterate.sh run    [models/<model>.env]   # start, or carry on where it stopped
#   bash abliterate.sh status [models/<model>.env]   # which stages are done, and the tables
#   bash abliterate.sh redo <stage> [models/<model>.env]   # forget a stage and every later one
#   bash abliterate.sh plan   [models/<model>.env]   # print what `run` would do; changes nothing
#
# The stages, in order. Each leaves a marker on the head node when it has
# finished, so `run` can be interrupted and started again: finished stages are
# skipped, and the long ones carry on by themselves on the head node meanwhile.
# (The machine running this must stay awake while a stage waits, or the
# pipeline pauses with a scratch build still served.)
#
#   preflight       weights, images, profiles and data are where they need to
#                   be; the quantized formats are ones the tools reproduce exactly
#   directions      measure the refusal direction at every layer, in the lab
#                   (serving stops; ~30 min)
#   screen          build / serve / collect a ladder of edits, plus an unedited
#                   control (~50 min each)
#   screen_judge    serve the stock model again; it judges what was collected
#   finalists       choose which go on: as good as the best, least drift
#   validate        held-out prompts, benign prompts, GSM8K, MMLU and the
#                   robustness scenarios, for each finalist (~2.5 h each)
#   validate_judge  as screen_judge
#   winner          choose one, by the same rule plus what it must not have lost
#   finalize        build it under its kept name, with ABLATION.txt, on both nodes
#   restore         put the cluster back to what it was serving
#
# THE CLUSTER SERVES SCRATCH BUILDS DURING directions, screen, validate AND
# finalize: the usual model is unavailable for hours. `run` says so and waits
# five seconds before the first stage that causes it.
#
# The judge is the stock model, served as before between the serving stages
# (JUDGE_MODE=deferred): a candidate occupies both nodes, so nothing else can
# run beside it. Settings come from the model's file (models/*.env) and the
# site's (site.env). The pieces are ordinary scripts and can be run by hand:
# lab.sh, queue.sh, finalize.sh, box/judge_pending.sh. See docs/reproducing.md.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$HERE/site.env" ]] || { echo "!! no site.env: cp site.env.example site.env and fill it in"; exit 2; }
# shellcheck source=site.env.example
source "$HERE/site.env"

ACTION="${1:-}"
REDO=""
[[ "$ACTION" == redo ]] && { REDO="${2:?usage: $0 redo <stage> [model.env]}"; shift; }
SOURCE_ENV="${2:-${MODEL_ENV:-$HERE/models/deepseek-v4-flash-0731.env}}"
[[ -f "$SOURCE_ENV" ]] || { echo "!! no such model file: $SOURCE_ENV"; exit 2; }
# shellcheck disable=SC1090
source "$SOURCE_ENV"

# This repository carries the paths the DeepSeek-V4 run used: a model served
# across both nodes, the lab on the head node, the stock model as the judge.
[[ "${JUDGE_MODE:-}" == deferred && "${TWO_BOX:-0}" == 1 && "${LAB_BOX:-}" == head ]] || {
  echo "!! this pipeline needs JUDGE_MODE=deferred, TWO_BOX=1 and LAB_BOX=head in $SOURCE_ENV"; exit 2; }

STAGES=(preflight directions screen screen_judge finalists validate validate_judge winner finalize restore)
RUN="${RUN:-$MODEL_KEY}"
PIPE="pipeline/$RUN"                       # under ~/abliterate on the head node
DIRECTION="${DIRECTION:-$RUN-d1}"
SWITCH="$SWITCH_MODEL"; [[ "$SWITCH" == /* ]] || SWITCH="$HERE/$SWITCH"

# Every script this calls reads the model's file. A pipeline run gets a copy
# with its own results directory, so runs do not mix with each other or with
# experiments started by hand.
mkdir -p "$HERE/.pipeline"
export MODEL_ENV="$HERE/.pipeline/$RUN.env"
{ cat "$SOURCE_ENV"; echo; echo "# added by abliterate.sh for run '$RUN'"; echo "EXP_ROOT=$PIPE/experiments"; } > "$MODEL_ENV"
EXP_ROOT="$PIPE/experiments"
STOCK="${STOCK_FIRST_TOKEN:-kl/stock.json}"

say()      { printf '\n==> [%s] %s\n' "$(date +%H:%M)" "$*"; }
head_box() { ssh "$HEAD_BOX" "$@"; }
work_box() { ssh "$WORK_BOX" "$@"; }
finished() { head_box "test -e ~/abliterate/$PIPE/state/$1"; }
mark()     { head_box "mkdir -p ~/abliterate/$PIPE/state && date -Is > ~/abliterate/$PIPE/state/$1"; }
recall()   { head_box "cat ~/abliterate/$PIPE/state/$1 2>/dev/null" || true; }
remember() { head_box "mkdir -p ~/abliterate/$PIPE/state && cat > ~/abliterate/$PIPE/state/$1" <<<"$2"; }
need()     { "${@:2}" || { echo "!! $1"; exit 1; }; }
# What the head node's run_queue.sh / judge_pending.sh need besides the model's file.
box_env()  { cat "$MODEL_ENV"; echo "BASE_IMAGE=$BASE_IMAGE"; echo "SERVE_DIR=$SERVE_DIR"; echo "SERVE_UNIT=$SERVE_UNIT"; echo "COPY_HOSTS=$COPY_HOSTS"; }

layers() {
  # Where the edit goes. LADDER_LAYERS overrides; then the band tools/geometry.py
  # suggested when the directions were measured; then a default.
  if [[ -n "${LADDER_LAYERS:-}" ]]; then echo "$LADDER_LAYERS"; return; fi
  local band; band="$(recall layer-band)"
  if [[ -n "$band" ]]; then echo "$band"; return; fi
  local n
  n="$(head_box "grep -o '\"num_hidden_layers\": *[0-9]*' ~/models/$BF16_NAME/config.json | head -n 1 | grep -o '[0-9]*\$'")"
  echo "$(( (n * 3 + 5) / 10 ))-$(( n - 1 ))"
}

# --- stages --------------------------------------------------------------------

stage_preflight() {
  need "cannot reach $HEAD_BOX" head_box true
  need "cannot reach $WORK_BOX" work_box true
  # A relative path: on Windows, Python does not understand Git Bash's /c/... form.
  (cd "$HERE" && python3 -c "import sys; sys.path.insert(0, 'tools'); import arch; arch.get('$ARCH')")

  need "stock weights missing on $HEAD_BOX: ~/models/$STOCK_NAME"  head_box "test -f ~/models/$STOCK_NAME/config.json"
  need "stock weights missing on $WORK_BOX: ~/models/$STOCK_NAME"  work_box "test -f ~/models/$STOCK_NAME/config.json"
  need "image $BASE_IMAGE missing on $HEAD_BOX"                     head_box "docker image inspect $BASE_IMAGE >/dev/null 2>&1"
  need "image ${LAB_IMAGE:-$IMAGE} missing on $HEAD_BOX"            head_box "docker image inspect ${LAB_IMAGE:-$IMAGE} >/dev/null 2>&1"
  local p
  for p in "$EXP_PROFILE" "$KEPT_PROFILE"; do
    need "no serving profile '$p' on $HEAD_BOX (copy serving/profiles/$p.env to $SERVE_DIR/models/)" \
      head_box "test -f $SERVE_DIR/models/$p.env"
  done
  if head_box "test -e ~/models/$KEPT_NAME" && [[ "${FORCE:-0}" != "1" ]]; then
    echo "!! ~/models/$KEPT_NAME already exists on $HEAD_BOX. This run would replace it:"
    echo "   set KEPT_NAME to a new name in $SOURCE_ENV, or run with FORCE=1."; exit 1
  fi

  # What to put back at the end. Recorded once: a later run of this stage would
  # find the scratch profile there.
  local serving; serving="$(head_box "cat $SERVE_DIR/active-model")"
  if [[ -z "$(recall restore-profile)" ]]; then
    serving="${RESTORE_PROFILE:-$serving}"
    if [[ "$serving" == "$EXP_PROFILE" ]]; then
      echo "!! The cluster is serving the scratch profile, so there is nothing sensible to restore to."
      echo "   Say what to put back at the end:  RESTORE_PROFILE=<profile> bash abliterate.sh run"; exit 1
    fi
    remember restore-profile "$serving"
  fi
  echo "    will restore the cluster to: $(recall restore-profile)"
  # The judge is whatever the restored profile serves; it has to be the stock model.
  local served
  served="$(head_box "grep -h '^SERVED_NAME=' $SERVE_DIR/models/$(recall restore-profile).env | tail -n 1 | cut -d= -f2")"
  [[ "$served" == "$JUDGE_MODEL" ]] || {
    echo "!! The judge is what '$(recall restore-profile)' serves ('$served'),"
    echo "   but JUDGE_MODEL is '$JUDGE_MODEL'. Restore to the stock model's profile."; exit 1; }
  echo "    judge: $JUDGE_MODEL, served between the serving stages"

  # Prompts and tests travel with the repository; the head node gets what it lacks.
  # The harmful prompt sets are built, not shipped: data/README.md.
  for f in $HARMFUL_PROMPTS $QUICK_PROMPTS $VALIDATION_HARMFUL $ROBUST_PROMPTS; do
    need "missing data/$f (python3 tools/build_prompt_sets.py data/prompts; data/README.md)" test -s "$HERE/data/$f"
  done
  tar -C "$HERE/data" -cf - . | head_box "mkdir -p ~/abliterate && tar -C ~/abliterate --skip-old-files -xf -"
  head_box "mkdir -p ~/abliterate/tools ~/abliterate/box ~/abliterate/directions ~/abliterate/kl ~/abliterate/$EXP_ROOT"
  scp -q "$HERE"/tools/*.py "$HEAD_BOX":abliterate/tools/
  scp -q "$HERE"/box/*.sh "$HEAD_BOX":abliterate/box/

  echo "    checking the quantized formats against the stock build..."
  local verdict
  verdict="$(head_box "docker run --rm --user \$(id -u):\$(id -g) -e HOME=/tmp -v \$HOME/models:/models:ro -v \$HOME/abliterate:/work:ro \
      --entrypoint python3 $BASE_IMAGE /work/tools/check_formats.py --arch $ARCH /models/$BF16_NAME /models/$STOCK_NAME 2>&1 | tail -n 1")"
  echo "    $verdict"
  if [[ "$verdict" != *"all checked tensors: 1.000000" ]]; then
    echo "!! The encoders in tools/quant_formats.py do not reproduce this checkpoint's values exactly."
    echo "   Edits would be written in a slightly different format. See docs/quantization.md."; exit 1
  fi
}

stage_directions() {
  if head_box "test -s ~/abliterate/directions/$DIRECTION.safetensors"; then
    echo "    directions/$DIRECTION.safetensors already on $HEAD_BOX; using it"
  else
    directions_on_head
  fi
  # Where to edit: the band tools/geometry.py suggests, unless LADDER_LAYERS says.
  scp -q "$HERE"/tools/*.py "$HEAD_BOX":abliterate/tools/
  local report
  report="$(head_box "docker run --rm --user \$(id -u):\$(id -g) -e HOME=/tmp -v \$HOME/abliterate:/work:ro \
      --entrypoint python3 $BASE_IMAGE /work/tools/geometry.py /work/directions/$DIRECTION.safetensors 2>&1")"
  sed 's/^/    /' <<<"$report"
  if grep -q "COPIES DISAGREE" <<<"$report"; then
    if [[ "${HC_DISAGREE_OK:-0}" != "1" ]]; then
      echo "!! The hyper-connection copies carry different directions (see above): one direction"
      echo "   removed from all four is not the measured refusal. Stop and look before building."
      echo "   (A ladder that also removes the copies' span sets HC_DISAGREE_OK=1; docs/method.md.)"; exit 1
    fi
    echo "    copies disagree; per-copy directions written too (HC_DISAGREE_OK=1)"
    head_box "docker run --rm --user \$(id -u):\$(id -g) -e HOME=/tmp -v \$HOME/abliterate:/work \
        --entrypoint python3 $BASE_IMAGE /work/tools/split_copies.py /work/directions/$DIRECTION.safetensors" \
      | sed 's/^/    /'
  fi
  local band; band="$(sed -n 's/^suggested band[^:]*: \([0-9]*-[0-9]*\).*/\1/p' <<<"$report")"
  [[ -n "$band" ]] && remember layer-band "$band"
  echo "    edit layers: $(layers)"
}

directions_on_head() {
  # The model fills both nodes, and its weights are where the head node's GPU
  # is. First, while the stock model is still served, the server's own
  # first-token distributions for the lab's gate (lab.cli directions --gate);
  # then serving stops and the lab streams the checkpoint.
  local gate="directions/$DIRECTION-gate"
  # Kept once taken: a rerun after a failed lab finds serving already stopped.
  head_box "cd ~/abliterate && test -s $gate-vllm.json || { head -n ${GATE_PROMPTS:-20} $HARMLESS_PROMPTS > $gate-prompts.txt &&
    THINK_KWARG=${THINK_KWARG:-enable_thinking} python3 tools/first_token_kl.py dump --token-ids \
      --model ${STOCK_SERVED:-$JUDGE_MODEL} --prompts $gate-prompts.txt --out $gate-vllm.json; }"
  outage_notice
  head_box "systemctl --user stop $SERVE_UNIT"
  # Batches of 32: the per-expert loop in the MoE costs per batch, not per prompt.
  bash "$HERE/lab.sh" run "$DIRECTION" directions --stream --gpu-batch 32 --out "/directions/$DIRECTION.safetensors" \
    --gate "/directions/$DIRECTION-gate-vllm.json" --gate-out "/directions/$DIRECTION-gate-lab.json"
  bash "$HERE/lab.sh" wait "$DIRECTION" || { bash "$HERE/lab.sh" log "$DIRECTION" 12; exit 1; }
  bash "$HERE/lab.sh" log "$DIRECTION" 6
}

outage_notice() {
  [[ -n "$(recall outage-announced)" ]] && return
  echo
  echo "    From here the cluster serves scratch builds, not '$(recall restore-profile)',"
  echo "    until the restore stage. Ctrl-C in the next 5 seconds to stop."
  sleep 5
  remember outage-announced yes
}

run_queue() {  # run_queue <queue file> <marker each experiment must end with> [VAR=value ...]
  local queue="$1" marker="collected-$2"; shift 2   # judged afterwards, by the *_judge stage
  if ! head_box 'pgrep -f "[b]ox/run_queue.sh" >/dev/null'; then
    env "$@" bash "$HERE/queue.sh" start "$queue"
  else
    echo "    a queue is already running on $HEAD_BOX; waiting for it"
  fi
  bash "$HERE/queue.sh" wait
  local name missing=()
  while IFS='|' read -r name _; do
    name="${name//[[:space:]]/}"
    [[ -z "$name" || "$name" == \#* ]] && continue
    # </dev/null: ssh would otherwise read the rest of the queue file as its
    # stdin, and only the first experiment would ever be checked.
    head_box "test -e ~/abliterate/$EXP_ROOT/$name/$marker" </dev/null || missing+=("$name")
  done < "$queue"
  if (( ${#missing[@]} )); then
    echo "!! did not finish: ${missing[*]}. ssh $HEAD_BOX tail -n 30 '~/abliterate/queue.log'"
    echo "   Run this again to retry them; finished experiments are kept."; exit 1
  fi
}

stage_screen() {
  outage_notice
  local range queue="$HERE/.pipeline/$RUN-screen.txt"
  range="$(layers)"
  {
    echo "control | --directions $DIRECTION --layers $range --weight 0"
    # The model's own rungs: "name | ablate.py flags", one per line.
    echo "# Written by abliterate.sh: one direction, layers $range, the rungs in LADDER."
    local name rest
    while IFS='|' read -r name rest; do
      name="${name//[[:space:]]/}"; [[ -z "$name" ]] && continue
      # A rung naming its own --directions uses those instead of $DIRECTION.
      if [[ "$rest" == *--directions* ]]; then echo "$name | --layers $range $rest"
      else echo "$name | --directions $DIRECTION --layers $range $rest"; fi
    done <<<"$LADDER"
  } > "$queue"
  sed 's/^/    /' "$queue"
  run_queue "$queue" done QUICK="${SCREEN_QUICK:-0}"

  # Drift is measured from the stock model. The control is the stock model
  # (an edit of weight 0, re-encoded), so it supplies the reference if there
  # is none yet.
  head_box "cd ~/abliterate && { test -s $STOCK || { mkdir -p \$(dirname $STOCK) && cp $EXP_ROOT/control/first-token.json $STOCK; }; }"
}

judge_deferred() {
  # Put the stock model back (it is what was being served, and the judge), then
  # judge everything the last serving stage collected.
  local profile; profile="$(recall restore-profile)"
  if [[ "$(head_box "cat $SERVE_DIR/active-model")" != "$profile" ]] \
      || ! head_box "curl -sf -m 5 ${JUDGE_BASE%/}/models | grep -q '\"$JUDGE_MODEL\"'"; then
    bash "$SWITCH" "$profile"
  fi
  if ! head_box 'pgrep -f "[b]ox/judge_pending.sh" >/dev/null'; then
    scp -q "$HERE"/tools/*.py "$HEAD_BOX":abliterate/tools/
    scp -q "$HERE"/box/*.sh "$HEAD_BOX":abliterate/box/
    box_env | head_box 'cat > ~/abliterate/model.env'
    head_box "cd ~/abliterate && (setsid nohup bash box/judge_pending.sh ~/abliterate/model.env >> judge-pending.log 2>&1 < /dev/null &)"
  else
    echo "    judging is already running on $HEAD_BOX; waiting for it"
  fi
  local last="" line
  while head_box 'pgrep -f "[b]ox/judge_pending.sh" >/dev/null'; do
    line="$(head_box 'tail -n 1 ~/abliterate/judge-pending.log' | cut -c1-110)"
    [[ "$line" != "$last" ]] && { echo "    $line"; last="$line"; }
    sleep 60
  done
  if head_box "ls ~/abliterate/$EXP_ROOT/*/pending-judge >/dev/null 2>&1"; then
    echo "!! some responses were not judged: ssh $HEAD_BOX tail -n 30 '~/abliterate/judge-pending.log'"
    echo "   Run this again to retry; verdicts already written are kept."; exit 1
  fi
}

stage_screen_judge()   { judge_deferred; }
stage_validate_judge() { judge_deferred; }

stage_finalists() {
  head_box "cd ~/abliterate && python3 tools/choose.py screen $EXP_ROOT --stock $STOCK \
      --tolerance ${SELECT_TOLERANCE:-3} --keep ${FINALISTS:-2} --write-queue $PIPE/finalists.txt"
  head_box "cat ~/abliterate/$PIPE/finalists.txt" > "$HERE/.pipeline/$RUN-finalists.txt"
}

stage_validate() {
  outage_notice
  [[ -s "$HERE/.pipeline/$RUN-finalists.txt" ]] || head_box "cat ~/abliterate/$PIPE/finalists.txt" > "$HERE/.pipeline/$RUN-finalists.txt"
  run_queue "$HERE/.pipeline/$RUN-finalists.txt" validated VALIDATE=1
}

stage_winner() {
  local out
  out="$(head_box "cd ~/abliterate && python3 tools/choose.py winner $EXP_ROOT --stock $STOCK \
      --tolerance ${SELECT_TOLERANCE:-3} --mmlu-drop ${MMLU_DROP:-5} --min-benign ${MIN_BENIGN:-95} \
      --max-unfinished ${MAX_UNFINISHED:-2} --min-speed ${MIN_SPEED:-0}")" || { echo "$out"; echo
    echo "!! No finalist passed. Nothing was built. The tables: bash abliterate.sh status"
    echo "   Loosen a limit in $SOURCE_ENV and \`redo winner\`, or widen the ladder and \`redo screen\`."; exit 1; }
  echo "$out"
  remember winner-name "$(sed -n 's/^WINNER //p' <<<"$out")"
}

stage_finalize() {
  local winner; winner="$(recall winner-name)"
  need "no winner recorded; bash abliterate.sh redo winner" test -n "$winner"
  # The build decodes and re-encodes on the GPU the served model holds, and the
  # kept build goes to both nodes: a short outage, ended by `restore`.
  outage_notice
  head_box "systemctl --user stop $SERVE_UNIT"
  FORCE="${FORCE:-0}" COPY_TO_WORK=1 bash "$HERE/finalize.sh" "$winner"
}

stage_restore() {
  local profile; profile="$(recall restore-profile)"
  [[ "${SERVE_KEPT:-0}" == "1" ]] && profile="$KEPT_PROFILE"
  bash "$SWITCH" "$profile"
  echo
  echo "    Built: ~/models/$KEPT_NAME on both nodes (ABLATION.txt inside says how, and how it scored)."
  [[ "${SERVE_KEPT:-0}" == "1" ]] || echo "    To serve it: bash $SWITCH $KEPT_PROFILE"
}

# --- commands ------------------------------------------------------------------

case "$ACTION" in
  run)
    for stage in "${STAGES[@]}"; do
      if finished "$stage"; then echo "    $stage: done $(recall "$stage")"; continue; fi
      say "$stage"
      # If a stage fails the cluster may be left serving a scratch build; say how to get out.
      trap 'echo; echo "!! stopped in stage: '"$stage"'. Fix it and run this again, or put the cluster back with:"; echo "   bash '"$SWITCH"' $(recall restore-profile)"' ERR
      "stage_$stage"
      trap - ERR
      mark "$stage"
    done
    say "finished"
    ;;
  status)
    for stage in "${STAGES[@]}"; do printf '  %-14s %s\n' "$stage" "$(recall "$stage" || true)"; done
    echo "  restore to: $(recall restore-profile)   winner: $(recall winner-name)"
    echo
    bash "$HERE/queue.sh" status 2>/dev/null || true
    echo
    bash "$HERE/queue.sh" results --validation 2>/dev/null || true
    ;;
  redo)
    seen=0
    for stage in "${STAGES[@]}"; do
      [[ "$stage" == "$REDO" ]] && seen=1
      (( seen )) && head_box "rm -f ~/abliterate/$PIPE/state/$stage"
    done
    (( seen )) || { echo "!! no stage '$REDO'. Stages: ${STAGES[*]}"; exit 2; }
    echo "forgot $REDO and every later stage. Results already collected are kept and reused."
    ;;
  plan)
    echo "model file   $SOURCE_ENV   (run '$RUN')"
    echo "edit         $STOCK_NAME, decoded from its own FP8/FP4 weights"
    echo "direction    $DIRECTION   layers $(layers)   rungs: control, $(cut -d'|' -f1 <<<"$LADDER" | xargs | sed 's/ /, /g')"
    echo "judge        $JUDGE_MODEL, served between the serving stages"
    echo "choose       within ${SELECT_TOLERANCE:-3} points of the best; then least drift; ${FINALISTS:-2} finalists"
    echo "must keep    MMLU within ${MMLU_DROP:-5} of the control, benign >= ${MIN_BENIGN:-95}%, unfinished reasoning <= ${MAX_UNFINISHED:-2}%, speed >= ${MIN_SPEED:-0} x control"
    echo "result       ~/models/$KEPT_NAME on both nodes, served by profile $KEPT_PROFILE"
    echo "results in   ~/abliterate/$EXP_ROOT on $HEAD_BOX"
    for stage in "${STAGES[@]}"; do printf '  %-14s %s\n' "$stage" "$(recall "$stage" || true)"; done
    ;;
  *)
    sed -n '2,38p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2
    ;;
esac
