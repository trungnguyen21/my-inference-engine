#!/usr/bin/env bash
# Push source to the EC2 node(s) and optionally build.
#
# Source only: the model is downloaded on the instance by the bootstrap script,
# and build/ is machine-specific. That keeps each sync to well under a megabyte
# instead of half a gigabyte.
#
#   scripts/sync.sh            # sync to every node
#   scripts/sync.sh --build    # sync, then build the engine and the server
#   scripts/sync.sh --node 1   # just node-1 (1-indexed)
set -euo pipefail

cd "$(dirname "$0")/.."
REPO=$PWD
BUILD=0
ONLY=""

while [ $# -gt 0 ]; do
  case $1 in
    --build) BUILD=1 ;;
    --node)  ONLY=$2; shift ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
  shift
done

# Portable array fill: macOS ships bash 3.2, which has no mapfile.
IPS=()
while IFS= read -r line; do
  [ -n "$line" ] && IPS+=("$line")
done < <(cd infra && terraform output -json node_ips \
         | python3 -c 'import json,sys; print("\n".join(json.load(sys.stdin)))')

if [ ${#IPS[@]} -eq 0 ]; then
  echo "no nodes: run 'cd infra && terraform apply' first" >&2
  exit 1
fi

for i in "${!IPS[@]}"; do
  n=$((i + 1))
  [ -n "$ONLY" ] && [ "$ONLY" != "$n" ] && continue
  ip=${IPS[$i]}
  echo "=== node-$n ($ip) ==="

  if ! ssh -o ConnectTimeout=5 "ubuntu@$ip" 'test -f /opt/engine/BOOTSTRAP_DONE' 2>/dev/null; then
    echo "  bootstrap still running; tail /var/log/bootstrap.log on the node" >&2
    continue
  fi

  rsync -az --delete \
    --exclude 'build/' --exclude 'models/' --exclude '.git/' \
    --exclude 'docs/' --exclude 'media/' --exclude 'infra/.terraform*' \
    --exclude 'infra/*.tfstate*' --exclude 'server/server' \
    "$REPO/" "ubuntu@$ip:/opt/engine/repo/"
  echo "  synced"

  if [ "$BUILD" = 1 ]; then
    ssh "ubuntu@$ip" 'bash -s' <<'REMOTE'
set -euo pipefail
source /etc/profile.d/go.sh
cd /opt/engine/repo
# The engine target only; the test binaries are not needed to serve.
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENGINE_OPENMP=ON >/dev/null
cmake --build build --target engine -j"$(nproc)" 2>&1 | tail -3
cd server
go build -o /opt/engine/server . && echo "  server built: $(ls -lh /opt/engine/server | awk '{print $5}')"
REMOTE
  fi
done
