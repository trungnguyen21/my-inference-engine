#!/usr/bin/env bash
# Kill something, then watch the cluster heal — with timestamps, so the
# recovery story has numbers.
#
#   ./deploy/chaos.sh pod          graceful delete: SIGTERM, the 65 s drain
#                                  applies, in-flight generations finish.
#                                  Expect ZERO client impact — that is the
#                                  point of this mode.
#   ./deploy/chaos.sh pod --hard   grace-period=0: the process is killed
#                                  mid-token. Streams break; this is the mode
#                                  that exercises client retries.
#   ./deploy/chaos.sh node         terminates the victim pod's EC2 INSTANCE.
#                                  The big one: node goes NotReady, pods
#                                  reschedule, the ASG boots a replacement.
#                                  Asks for confirmation first.
#
# Always kills a pod on the BUSIEST node when it can (most engine pods), so
# the demo shows rebalancing too. Run the dashboard + loadgen while this runs.
set -euo pipefail

MODE=${1:-pod}
HARD=${2:-}

desired=$(kubectl get deploy engine -o jsonpath='{.spec.replicas}')

# Victim: a Running engine pod on the node that currently has the most of
# them (awk builds node->count, then we take a pod from the max node).
victim=$(kubectl get pods -l app=engine --field-selector=status.phase=Running \
  -o custom-columns=POD:.metadata.name,NODE:.spec.nodeName --no-headers |
  awk '{n[$2]++; pod[$2]=$1} END {best=""; for (k in n) if (n[k]>n[best] || best=="") best=k; print pod[best], best}')
pod=${victim% *}
node=${victim#* }

echo "== desired replicas: $desired; victim pod: $pod (on $node)"

start=$(date +%s)
stamp() { printf '[%3ds] %s\n' "$(( $(date +%s) - start ))" "$1"; }

case "$MODE" in
  pod)
    if [ "$HARD" = "--hard" ]; then
      stamp "HARD-killing $pod (no drain, streams on it will break)"
      kubectl delete pod "$pod" --grace-period=0 --force >/dev/null
    else
      stamp "gracefully deleting $pod (watch it linger Terminating while it drains)"
      kubectl delete pod "$pod" --wait=false >/dev/null
    fi
    ;;
  node)
    iid=$(aws ec2 describe-instances --region ca-central-1 \
      --filters "Name=private-dns-name,Values=$node" "Name=instance-state-name,Values=running" \
      --query 'Reservations[0].Instances[0].InstanceId' --output text)
    echo "!! about to TERMINATE EC2 instance $iid ($node)."
    echo "!! make sure your browser/loadgen points at the OTHER node's IP."
    read -r -p "type the word kill to proceed: " ok
    [ "$ok" = "kill" ] || { echo "aborted"; exit 1; }
    stamp "terminating $iid; expect ~40 s to NotReady, ~5 min for pod eviction, more for the ASG replacement"
    aws ec2 terminate-instances --instance-ids "$iid" --region ca-central-1 >/dev/null
    ;;
  *)
    echo "usage: $0 [pod [--hard] | node]"; exit 1 ;;
esac

# Watch: print a line only when the picture changes, so the log reads as a
# timeline of the heal rather than a wall of identical polls.
last=""
while true; do
  snap=$(kubectl get pods -l app=engine \
    -o custom-columns=POD:.metadata.name,STATUS:.status.phase,READY:'.status.conditions[?(@.type=="Ready")].status',NODE:.spec.nodeName \
    --no-headers 2>/dev/null | sort)
  ready=$(kubectl get deploy engine -o jsonpath='{.status.readyReplicas}' 2>/dev/null)
  ready=${ready:-0}
  if [ "$snap" != "$last" ]; then
    stamp "ready: $ready/$desired"
    echo "$snap" | sed 's/^/        /'
    last=$snap
  fi
  if [ "$ready" = "$desired" ] && ! echo "$snap" | grep -q "$pod"; then
    stamp "healed: $desired/$desired ready and the victim is gone"
    break
  fi
  sleep 2
done
