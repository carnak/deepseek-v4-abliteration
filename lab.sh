#!/usr/bin/env bash
# Run a lab command on the lab node (LAB_BOX in the model's file; the head node
# for DeepSeek-V4), detached, in a container.
#
#   bash lab.sh run <name> <lab.cli arguments...>    # start (container lab-<name>)
#   bash lab.sh log <name> [lines]                   # tail it
#   bash lab.sh stop <name>
#   bash lab.sh wait <name>                          # block until it exits; fails if it failed
#   bash lab.sh fetch <file>                         # copy a directions file work node -> head node
#
#   bash lab.sh run deepseek-v4-nothink directions --stream --gpu-batch 32 \
#       --out /directions/deepseek-v4-nothink.safetensors
#   bash lab.sh run screen dsv4-screen --config mean2.5-13-35:deepseek-v4-nothink:13-35:2.5
#
# --model and --prompts are supplied for you. Inside the container:
#   /models/<MODEL>   the stock weights (read-only)
#   /prompts          ~/abliterate/prompts (read-only)
#   /directions       ~/abliterate/directions, where direction files are read and written
#
# The lab needs the GPU to itself: it refuses to start while a serving container
# or another lab run is up there (abliterate.sh stops serving first).
#
# The code runs from this checkout: it is copied to the box on every start, so
# there is no image to rebuild after an edit. The image only supplies PyTorch.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=site.env.example
source "$HERE/site.env"
MODEL_ENV="${MODEL_ENV:-$HERE/models/deepseek-v4-flash-0731.env}"
# shellcheck disable=SC1090
source "$MODEL_ENV"
# Where the lab runs (LAB_BOX in the model's file): the work box by default; the
# head box for a model whose weights only it has room for, or whose lab needs
# the serving image (LAB_IMAGE; DeepSeek-V4 needs its transformers 5.19).
if [[ "${LAB_BOX:-work}" == "head" ]]; then LAB_HOST="$HEAD_BOX"; else LAB_HOST="$WORK_BOX"; fi
LAB_IMAGE="${LAB_IMAGE:-$IMAGE}"

ACTION="${1:-}"
NAME="${2:-}"
[[ -n "$ACTION" && -n "$NAME" ]] || { sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2; }
shift 2
CONTAINER="lab-$NAME"

case "$ACTION" in
  run)
    if ssh "$LAB_HOST" "docker ps --format '{{.Names}}' | grep -qE '^(vllm_node|vllm_judge|lab-.*|heretic-.*)\$'"; then
      echo "!! $LAB_HOST is busy:"; ssh "$LAB_HOST" "docker ps --format '   {{.Names}}  {{.Status}}'"; exit 1
    fi
    ssh "$LAB_HOST" "mkdir -p $WORK_DIR/lab $WORK_DIR/directions $WORK_DIR/prompts $WORK_DIR/code"
    # tar over ssh, not rsync: Git Bash on Windows does not ship rsync.
    tar -C "$HERE" --exclude __pycache__ -cf - lab tools | ssh "$LAB_HOST" "rm -rf $WORK_DIR/code/lab $WORK_DIR/code/tools && tar -C $WORK_DIR/code -xf -"
    ARGS="$(printf '%q ' "$@")"
    # The first argument after the name is the lab.cli subcommand.
    SUB="$1"; shift
    REST="$(printf '%q ' "$@")"
    ssh "$LAB_HOST" "docker rm $CONTAINER >/dev/null 2>&1 || true
      docker run -d --name $CONTAINER --gpus all --ipc host \
        --user \$(id -u):\$(id -g) -e HOME=/tmp -e PYTHONUNBUFFERED=1 -e PYTHONPATH=/code \
        -v \$HOME/models/$BF16_NAME:/models/$BF16_NAME:ro \
        -v \$(eval echo $WORK_DIR/code):/code:ro \
        -v \$(eval echo $WORK_DIR/prompts):/prompts:ro \
        -v \$(eval echo $WORK_DIR/directions):/directions \
        --entrypoint python3 $LAB_IMAGE -m lab.cli $SUB --model /models/$BF16_NAME --prompts /prompts $REST >/dev/null"
    echo "==> started $CONTAINER on $LAB_HOST: lab.cli $ARGS"
    echo "    bash lab.sh log $NAME"
    ;;
  log)
    LINES="${1:-30}"
    ssh "$LAB_HOST" "docker ps -a --filter name=^/$CONTAINER\$ --format '{{.Names}}  {{.Status}}'
      docker logs $CONTAINER 2>&1 | tr '\r' '\n' | grep -vE '^\s*\$|Loading weights|Warning' | tail -n $LINES"
    ;;
  stop)
    ssh "$LAB_HOST" "docker stop -t 5 $CONTAINER >/dev/null 2>&1; docker ps -a --filter name=^/$CONTAINER\$ --format '{{.Names}}  {{.Status}}'"
    ;;
  wait)
    while :; do
      state="$(ssh -o ConnectTimeout=10 "$LAB_HOST" "docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' $CONTAINER" 2>/dev/null || echo unreachable)"
      case "$state" in
        "exited 0") exit 0 ;;
        exited*|dead*) echo "!! $CONTAINER failed ($state). bash lab.sh log $NAME"; exit 1 ;;
      esac
      sleep 30
    done
    ;;
  fetch)
    # Directions are measured on the work box and applied on the head box.
    ssh "$HEAD_BOX" "mkdir -p ~/abliterate/directions"
    ssh "$WORK_BOX" "cat $WORK_DIR/directions/$NAME" | ssh "$HEAD_BOX" "cat > ~/abliterate/directions/$NAME"
    ssh "$HEAD_BOX" "ls -la ~/abliterate/directions/$NAME"
    ;;
  *) echo "usage: $0 run|log|stop|wait <name> ... | fetch <file>"; exit 2 ;;
esac
