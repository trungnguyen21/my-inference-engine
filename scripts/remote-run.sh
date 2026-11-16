#!/usr/bin/env bash
# Install/restart the server as a systemd unit on a node.
#
# systemd rather than nohup: a backgrounded process holds the SSH channel open,
# so `ssh node 'nohup ... &'` hangs. systemd also restarts on failure and keeps
# logs in journalctl, which is what the demo needs anyway.
#
#   scripts/remote-run.sh                       # defaults
#   scripts/remote-run.sh --workers 4 --context 1024
set -euo pipefail
cd "$(dirname "$0")/.."

WORKERS=2
CONTEXT=512
NODE=1

while [ $# -gt 0 ]; do
  case $1 in
    --workers) WORKERS=$2; shift ;;
    --context) CONTEXT=$2; shift ;;
    --node)    NODE=$2; shift ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
  shift
done

IP=$(cd infra && terraform output -json node_ips \
     | python3 -c "import json,sys; print(json.load(sys.stdin)[$NODE-1])")

ssh "ubuntu@$IP" "sudo tee /etc/systemd/system/inference.service >/dev/null" <<UNIT
[Unit]
Description=Inference engine HTTP server
After=network-online.target

[Service]
User=ubuntu
WorkingDirectory=/opt/engine
# POD_NAME/NODE_NAME stand in for what the Kubernetes Downward API will supply,
# so the demo kit sees the same fields here as it will in the cluster.
Environment=POD_NAME=%H
Environment=NODE_NAME=node-$NODE
Environment=MODEL=/opt/models/qwen-2.5.gguf
Environment=ADDR=:8080
Environment=WORKERS=$WORKERS
Environment=CONTEXT=$CONTEXT
ExecStart=/opt/engine/server
Restart=on-failure
RestartSec=2
# Match the Kubernetes terminationGracePeriod so draining behaves the same.
TimeoutStopSec=65
KillSignal=SIGTERM

[Install]
WantedBy=multi-user.target
UNIT

# Kill strays from any manual launch before systemd tries to bind the port.
# -x matches the process NAME exactly, so it cannot match this ssh command's
# own argv the way `pkill -f /opt/engine/server` would.
ssh "ubuntu@$IP" 'pkill -x server 2>/dev/null; sleep 1; true'

ssh "ubuntu@$IP" 'sudo systemctl daemon-reload && sudo systemctl enable --now inference.service && sudo systemctl restart inference.service'

# Check the UNIT is running, not merely that something answers on the port:
# a stale process would satisfy a port check while the new unit sat dead.
if ! ssh "ubuntu@$IP" 'sudo systemctl is-active --quiet inference'; then
  echo "unit is not active:" >&2
  ssh "ubuntu@$IP" 'sudo systemctl status inference --no-pager -n 20'
  exit 1
fi

echo "waiting for readiness on $IP ..."
for i in $(seq 1 30); do
  if curl -fsS --max-time 3 "http://$IP:8080/health/ready" >/dev/null 2>&1; then
    echo "ready after ~${i}s"
    curl -s "http://$IP:8080/health" | python3 -m json.tool
    exit 0
  fi
  sleep 1
done
echo "did not become ready; recent logs:" >&2
ssh "ubuntu@$IP" 'sudo journalctl -u inference -n 25 --no-pager'
exit 1
