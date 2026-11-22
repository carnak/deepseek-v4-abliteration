#!/usr/bin/env bash
# Runs ON THE HEAD BOX. Judges what a JUDGE_MODE=deferred queue collected.
#
#   judge_pending.sh <model.env>
#
# For a model that fills both boxes (DeepSeek-V4), the judge -- the stock model
# -- cannot run while a candidate is served, so box/run_queue.sh only collects
# and lists the response files in <experiment>/pending-judge. Once the stock
# model is served again (as production, at JUDGE_BASE), this judges every
# listed file and writes the markers a live queue would have written: "done",
# and "validated" for a validation run. Re-running is safe: an experiment
# leaves the pending list only once all of its files are judged.
#
# Started by ../abliterate.sh under nohup. Log: ~/abliterate/judge-pending.log.
set -uo pipefail

ROOT="$HOME/abliterate"
# shellcheck disable=SC1090
source "$1"
export THINK_KWARG="${THINK_KWARG:-enable_thinking}"
EXP_ROOT="${EXP_ROOT:-experiments}"
say() { printf '[%(%F %T)T] %s\n' -1 "$*"; }

cd "$ROOT" || exit 1
id="$(curl -s -m 10 "${JUDGE_BASE%/}/models" | grep -oE '"id":"[^"]*"' | head -1 | cut -d'"' -f4)"
if [[ "$id" != "$JUDGE_MODEL" ]]; then
  say "!! $JUDGE_BASE serves '${id:-nothing}', not the judge $JUDGE_MODEL"; exit 1
fi

failed=0
for pending in "$EXP_ROOT"/*/pending-judge; do
  [[ -e "$pending" ]] || continue
  dir="$(dirname "$pending")"; name="$(basename "$dir")"
  ok=1
  while read -r f; do
    [[ -z "$f" ]] && continue
    say "$name: judge $f"
    python3 tools/judge.py "$dir/$f.jsonl" --judge-base "$JUDGE_BASE" --judge-model "$JUDGE_MODEL" \
      --max-tokens "${JUDGE_MAX_TOKENS:-3000}" --concurrency "${JUDGE_CONCURRENCY:-8}" \
      > "$dir/judge-$f.txt" 2>&1 </dev/null || { ok=0; break; }
  done < <(sort -u "$pending")
  if (( ok )); then
    touch "$dir/done"
    [[ -e "$dir/collected-validated" ]] && touch "$dir/validated"
    rm -f "$pending"
    say "$name: judged ($(for f in "$dir"/judge-*.txt; do printf '%s [%s] ' "$(basename "$f" .txt)" "$(cut -d: -f2- "$f" | xargs)"; done))"
  else
    say "$name: JUDGE FAILED, see $dir/judge-$f.txt"; failed=1
  fi
done
say "=== judging finished"
exit $failed
