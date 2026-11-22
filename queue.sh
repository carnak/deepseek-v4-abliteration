#!/usr/bin/env bash
# Queue ablation experiments on the head box and leave them running.
#
#   bash queue.sh start <queue file>      # deploy tools, start the queue detached
#   VALIDATE=1 bash queue.sh start <file> # finalists: also the independent prompt
#                                         # set, benign prompts and GSM8K (~3x longer)
#   bash queue.sh results --validation    # the table for those
#   bash queue.sh results --robust        # and their robustness tier
#   QUICK=1 bash queue.sh start <file>    # screen on the 100-prompt tuning set
#   bash queue.sh status                  # log tail + results table
#   bash queue.sh results [name ...]      # results table only
#   bash queue.sh stop                    # stop after the current step
#   bash queue.sh wait                    # block until the queue has finished
#
# A queue file has one experiment per line:
#
#   <name> | <tools/ablate.py arguments>
#   m2.5-13-35 | --directions deepseek-v4-nothink --layers 13-35 --weight 2.5
#
# The queue runs on the box (box/run_queue.sh), not from this machine, so it
# survives this shell and anything that times it out. Each experiment takes
# about fifty minutes for this model; finished ones are skipped if the queue is
# started again.
#
# While a queue runs the cluster serves the scratch checkpoint, restarted for
# every experiment. Put it back afterwards with
#   bash serving/switch-model.sh <profile>   (abliterate.sh does this itself)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=site.env.example
source "$HERE/site.env"
MODEL_ENV="${MODEL_ENV:-$HERE/models/deepseek-v4-flash-0731.env}"
# shellcheck disable=SC1090
source "$MODEL_ENV"
EXP_ROOT="${EXP_ROOT:-experiments}"
STOCK="${STOCK_FIRST_TOKEN:-kl/stock.json}"

case "${1:-}" in
  start)
    QUEUE="${2:?usage: $0 start <queue file>}"
    ssh "$HEAD_BOX" "mkdir -p ~/abliterate/tools ~/abliterate/box ~/abliterate/$EXP_ROOT"
    if ssh "$HEAD_BOX" 'pgrep -f "[b]ox/run_queue.sh" >/dev/null'; then
      echo "!! a queue is already running on $HEAD_BOX. bash queue.sh status"; exit 1
    fi
    scp -q "$HERE"/tools/*.py "$HEAD_BOX":abliterate/tools/
    scp -q "$HERE/box/run_queue.sh" "$HEAD_BOX":abliterate/box/
    # The image the builds run in travels with the model's settings.
    { cat "$MODEL_ENV"; echo "BASE_IMAGE=$BASE_IMAGE"; echo "SERVE_DIR=$SERVE_DIR"; echo "SERVE_UNIT=$SERVE_UNIT"; echo "COPY_HOSTS=$COPY_HOSTS"; } | ssh "$HEAD_BOX" 'cat > ~/abliterate/model.env'
    scp -q "$QUEUE" "$HEAD_BOX":abliterate/queue.txt
    # setsid + nohup + closed stdin: fully detached, so ssh returns at once.
    ssh "$HEAD_BOX" "cd ~/abliterate && (VALIDATE=${VALIDATE:-0} QUICK=${QUICK:-0} setsid nohup bash box/run_queue.sh ~/abliterate/model.env ~/abliterate/queue.txt >> queue.log 2>&1 < /dev/null &)"
    echo "==> queue started on $HEAD_BOX: $(grep -cvE '^\s*(#|$)' "$QUEUE") experiment(s). bash queue.sh status"
    ;;
  status)
    ssh "$HEAD_BOX" 'cd ~/abliterate && (pgrep -f "[b]ox/run_queue.sh" >/dev/null && echo "queue: RUNNING" || echo "queue: not running"); tail -n 8 queue.log; echo; python3 tools/results.py --stock '"$STOCK $EXP_ROOT"
    ;;
  results)
    shift
    FLAGS=()
    if [[ "${1:-}" == "--validation" || "${1:-}" == "--robust" ]]; then FLAGS=("$1"); shift; fi
    ssh "$HEAD_BOX" "cd ~/abliterate && python3 tools/results.py ${FLAGS[*]} --stock $STOCK $EXP_ROOT $*"
    ;;
  wait)
    # A dropped connection is not the queue finishing: only a clean "no" ends the wait.
    last=""
    while :; do
      state="$(ssh -o ConnectTimeout=10 "$HEAD_BOX" 'pgrep -f "[b]ox/run_queue.sh" >/dev/null && echo running || echo idle; tail -n 1 ~/abliterate/queue.log' 2>/dev/null || echo unreachable)"
      line="$(tail -n 1 <<<"$state" | cut -c1-110)"
      [[ "$line" != "$last" ]] && { echo "    $line"; last="$line"; }
      [[ "$(head -n 1 <<<"$state")" == idle ]] && break
      sleep 60
    done
    ;;
  stop)
    ssh "$HEAD_BOX" 'pkill -f "[b]ox/run_queue.sh" && echo stopped || echo "no queue running"'
    ;;
  *)
    sed -n '2,23p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2
    ;;
esac
