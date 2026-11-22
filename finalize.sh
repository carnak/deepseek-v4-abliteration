#!/usr/bin/env bash
# Turn a finished experiment into the kept, named checkpoint.
#
#   bash finalize.sh <experiment>            # build it under KEPT_NAME on the head box
#   COPY_TO_WORK=1 bash finalize.sh <exp>    # ...and copy it to the work box, for TP=2
#
# An experiment's checkpoint is scratch: the next experiment overwrites it. This
# rebuilds the chosen one from the arguments recorded in its directory, into
# ~/models/$KEPT_NAME (models/<model>.env), and writes ABLATION.txt there: the
# exact command, where the directions came from, and the scores it was chosen
# on. The direction files themselves go into a directions/ subfolder, so the
# checkpoint can be reproduced from what is inside it.
#
# It does not change what the cluster serves. Afterwards:
#   bash serving/switch-model.sh $KEPT_PROFILE
#
# Refuses to overwrite an existing kept build unless FORCE=1: a kept build is
# the thing being served.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=site.env.example
source "$HERE/site.env"
MODEL_ENV="${MODEL_ENV:-$HERE/models/deepseek-v4-flash-0731.env}"
# shellcheck disable=SC1090
source "$MODEL_ENV"

EXP="${1:?usage: $0 <experiment>}"
EXP_ROOT="${EXP_ROOT:-experiments}"
STOCK="${STOCK_FIRST_TOKEN:-kl/stock.json}"

ssh "$HEAD_BOX" "test -f ~/abliterate/$EXP_ROOT/$EXP/ablate-args.txt" \
  || { echo "!! no experiment '$EXP' on $HEAD_BOX (bash queue.sh results)"; exit 1; }
if ssh "$HEAD_BOX" "test -e ~/models/$KEPT_NAME" && [[ "${FORCE:-0}" != "1" ]]; then
  echo "!! ~/models/$KEPT_NAME already exists on $HEAD_BOX. FORCE=1 to replace it."; exit 1
fi

scp -q "$HERE"/tools/*.py "$HEAD_BOX":abliterate/tools/

# A natively quantized model (DeepSeek-V4) is decoded, edited and re-encoded on
# the GPU; its build log ends with the edit-survival table, which ABLATION.txt keeps.
GPUS=""; [[ "${TWO_BOX:-0}" == "1" ]] && GPUS="--gpus all"

echo "==> $HEAD_BOX: build $EXP -> ~/models/$KEPT_NAME"
ssh "$HEAD_BOX" "set -e
  cd ~/abliterate
  ARGS=\"\$(cat $EXP_ROOT/$EXP/ablate-args.txt)\"
  # The recorded arguments name directions by short name, as queue files do.
  CARGS=() ; DIRS=() ; WEIGHTS=
  set -- \$ARGS
  while (( \$# )); do
    if [[ \"\$1\" == --directions ]]; then CARGS+=(--directions \"/work/directions/\$2.safetensors\"); DIRS+=(\"\$2\"); shift 2
    elif [[ \"\$1\" == --weights-file ]]; then CARGS+=(--weights-file \"/work/directions/\$2.json\"); WEIGHTS=\"\$2\"; shift 2
    else CARGS+=(\"\$1\"); shift; fi
  done
  rm -rf \$HOME/models/$KEPT_NAME
  docker run --rm $GPUS --user \$(id -u):\$(id -g) -e HOME=/tmp -v \$HOME/models:/models -v \$HOME/abliterate:/work:ro \
    --entrypoint python3 $BASE_IMAGE /work/tools/ablate.py --arch $ARCH --bf16 /models/$BF16_NAME \
    --quantized /models/$STOCK_NAME --out /models/$KEPT_NAME \"\${CARGS[@]}\" > /tmp/finalize-build.log 2>&1 \
    || { tail -n 20 /tmp/finalize-build.log; exit 1; }
  grep -E '^[0-9]+ direction|done|Error|Traceback' /tmp/finalize-build.log
  SURVIVAL=\"\$(sed -n '/^edit survival/,\$p' /tmp/finalize-build.log | grep -v '^done')\"
  [[ -n \"\$SURVIVAL\" ]] && echo \"\$SURVIVAL\"
  # A subfolder: vLLM would try to load any .safetensors beside the weights.
  mkdir -p \$HOME/models/$KEPT_NAME/directions
  for d in \"\${DIRS[@]}\"; do cp directions/\$d.safetensors \$HOME/models/$KEPT_NAME/directions/; done
  # Learned weights are part of the recipe: without them the checkpoint cannot be rebuilt.
  if [[ -n \"\$WEIGHTS\" ]]; then cp directions/\$WEIGHTS.json \$HOME/models/$KEPT_NAME/directions/; fi
  {
    echo \"$KEPT_NAME\"
    echo \"Abliterated build of $STOCK_NAME (original weights: $BF16_NAME).\"
    echo \"Built \$(date -Is) by deepseek-v4-abliteration/finalize.sh from experiment '$EXP':\"
    echo \"  tools/ablate.py --arch $ARCH \$ARGS\"
    echo
    echo 'Scores it was chosen on (percent: comply partial deflect refuse unfinished):'
    python3 tools/results.py --stock $STOCK $EXP_ROOT $EXP | sed 's/^/  /'
    python3 tools/results.py --validation --stock $STOCK $EXP_ROOT $EXP | sed 's/^/  /'
    python3 tools/results.py --robust $EXP_ROOT $EXP | sed 's/^/  /'
    [[ -s $EXP_ROOT/$EXP/speed.txt ]] && echo \"  decode speed: \$(cat $EXP_ROOT/$EXP/speed.txt)\"
    if [[ -n \"\$SURVIVAL\" ]]; then echo; echo \"\$SURVIVAL\"; fi
  } > \$HOME/models/$KEPT_NAME/ABLATION.txt
  cat \$HOME/models/$KEPT_NAME/ABLATION.txt"

if [[ "${COPY_TO_WORK:-0}" == "1" ]]; then
  if [[ "${TWO_BOX:-0}" == "1" ]]; then
    # The work box has room for one 155 GiB build beside the stock one, not two.
    echo "==> $WORK_BOX: remove the scratch build to make room"
    ssh "$WORK_BOX" "rm -rf ~/models/$SCRATCH_NAME"
  fi
  echo "==> $WORK_BOX: copy for TP=2"
  ssh "$WORK_BOX" "rsync -a --delete $HEAD_FAST_ADDR:models/$KEPT_NAME/ ~/models/$KEPT_NAME/ && du -sh ~/models/$KEPT_NAME | cut -f1"
fi

echo
echo "==> Kept build ready. Serve it with:"
if [[ "${TWO_BOX:-0}" == "1" ]]; then
  echo "    bash $SWITCH_MODEL $KEPT_PROFILE"
else
  echo "    bash $SWITCH_MODEL $KEPT_PROFILE-solo"
fi
