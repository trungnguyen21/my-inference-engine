#!/usr/bin/env bash
# Aggregate throughput, for comparing configurations.
#
# This drives loadgen rather than reimplementing it, so the request shape and
# the retry accounting stay identical to the traffic demo.
#
# The measurement that matters is one node versus two:
#
#   kubectl cordon node-2
#   demo/bench.sh -url http://NODE:30080 -label "1 node"
#   kubectl uncordon node-2
#   demo/bench.sh -url http://NODE:30080 -label "2 nodes"
#
# Expect roughly 2x. Memory bandwidth is per machine, so adding machines is the
# only change in this project that raises aggregate tokens/sec — adding workers
# on one box does not, which is measured in docs/serving-implementation-plan.md.
set -euo pipefail
cd "$(dirname "$0")"

URL="http://localhost:8080"
USERS=6
DURATION=60s
MAX_TOKENS=16
LABEL=""
WARMUP=10s

while [ $# -gt 0 ]; do
  case $1 in
    -url)        URL=$2; shift ;;
    -users)      USERS=$2; shift ;;
    -duration)   DURATION=$2; shift ;;
    -max-tokens) MAX_TOKENS=$2; shift ;;
    -label)      LABEL=$2; shift ;;
    -warmup)     WARMUP=$2; shift ;;
    -h|--help)
      sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
  shift
done

[ -n "$LABEL" ] || LABEL="$USERS users"

if ! curl -fsS --max-time 5 "$URL/health/ready" >/dev/null; then
  echo "not ready: $URL/health/ready" >&2
  exit 1
fi

BIN=$(mktemp -d)/loadgen
go build -o "$BIN" ./loadgen

# Warm up first. The first request on a fresh process pays for OpenMP team
# creation and for faulting the weights into the page cache, which would
# otherwise be charged to the measurement.
if [ "$WARMUP" != "0" ]; then
  echo "warming up for $WARMUP ..."
  "$BIN" -url "$URL" -users "$USERS" -max-tokens "$MAX_TOKENS" \
         -duration "$WARMUP" -refresh 1h >/dev/null 2>&1 || true
fi

echo "measuring for $DURATION ($USERS concurrent, max_tokens=$MAX_TOKENS) ..."
OUT=$("$BIN" -url "$URL" -users "$USERS" -max-tokens "$MAX_TOKENS" \
             -duration "$DURATION" -refresh 1h 2>&1)

# The summary lines look like:
#   ran 60s: 48 ok, 0 busy (retried), 0 failed, 384 tokens
#   1.28 req/s, 10.2 tok/s aggregate
RATES=$(printf '%s\n' "$OUT" | grep 'aggregate' || true)
COUNTS=$(printf '%s\n' "$OUT" | grep '^ran ' || true)
PODS=$(printf '%s\n' "$OUT" | awk '/^NODE/{f=1;next} f&&NF{n++} END{print n+0}')

echo
printf '%s\n' "$COUNTS"
printf '%s\n' "$RATES"
echo
printf '%-22s %-14s %s\n' "LABEL" "RESPONDERS" "AGGREGATE"
printf '%-22s %-14s %s\n' "$LABEL" "$PODS pod(s)" "${RATES:-no data}"
echo
echo "per-responder breakdown:"
printf '%s\n' "$OUT" | sed -n '/^NODE/,$p'
