#!/usr/bin/env bash
# Switch what the cluster serves, and wait until the new model answers.
#
#   bash serving/switch-model.sh <profile>
#
# A reference implementation of the one serving operation the pipeline needs,
# written against the arrangement site.env describes: on HEAD_BOX, the profile
# named in $SERVE_DIR/active-model is served by the systemd --user unit
# $SERVE_UNIT (across both nodes), on http://127.0.0.1:8001. It does what
# box/run_queue.sh's serve_scratch does from the head node itself.
#
# Replace it if your cluster is run differently: it must leave <profile>'s
# SERVED_NAME answering at /v1/models, and exit non-zero if it cannot.
#
# Switching is an outage: the old model stops before the new one loads.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../site.env.example
source "$HERE/site.env"
PROFILE="${1:?usage: $0 <profile>}"
TIMEOUT="${SERVE_TIMEOUT:-1500}"

served="$(ssh "$HEAD_BOX" "grep -h '^SERVED_NAME=' $SERVE_DIR/models/$PROFILE.env | tail -n 1 | cut -d= -f2")" \
  || { echo "!! no profile $SERVE_DIR/models/$PROFILE.env on $HEAD_BOX"; exit 1; }
echo "==> switching $HEAD_BOX to $PROFILE (serves '$served')"
ssh "$HEAD_BOX" "systemctl --user stop $SERVE_UNIT
  printf '%s\n' '$PROFILE' > $SERVE_DIR/active-model
  systemctl --user reset-failed $SERVE_UNIT 2>/dev/null || true
  systemctl --user start $SERVE_UNIT"

t0=$SECONDS
while (( SECONDS - t0 < TIMEOUT )); do
  id="$(ssh "$HEAD_BOX" "curl -s -m 5 http://127.0.0.1:8001/v1/models" 2>/dev/null \
        | grep -oE '"id":"[^"]*"' | head -n 1 | cut -d'"' -f4 || true)"
  if [[ "$id" == "$served" ]]; then echo "==> serving $served after $(( SECONDS - t0 )) s"; exit 0; fi
  if [[ "$(ssh "$HEAD_BOX" "systemctl --user is-active $SERVE_UNIT" || true)" == failed ]]; then
    echo "!! $SERVE_UNIT failed on $HEAD_BOX (journalctl --user -u $SERVE_UNIT)"; exit 1
  fi
  sleep 15
done
echo "!! $served did not answer within $TIMEOUT s"; exit 1
