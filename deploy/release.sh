#!/usr/bin/env bash
# Build, push, and roll out the engine image with an automatic tag.
#
# The tag is the git commit SHA, which is what CI systems do: it never
# repeats (so IfNotPresent can never serve a stale image), and `kubectl get
# pods -o yaml | grep image:` tells you exactly which commit is running.
# A dirty working tree gets a timestamp suffix, so uncommitted experiments
# still produce unique tags instead of silently reusing the last commit's.
#
# Usage, from anywhere in the repo:
#   ./deploy/release.sh
set -euo pipefail
cd "$(dirname "$0")/.."

ECR=327319899002.dkr.ecr.ca-central-1.amazonaws.com/inference-engine
REGION=ca-central-1
MANIFEST=deploy/k8s/eks/deployment.yaml

TAG=$(git rev-parse --short HEAD)
if [ -n "$(git status --porcelain -- . ':!deploy/k8s')" ]; then
  TAG="${TAG}-dirty-$(date +%H%M%S)"
fi

echo "==> building ${ECR}:${TAG}"
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$ECR" >/dev/null
docker build -t "${ECR}:${TAG}" .
docker push "${ECR}:${TAG}"

# Rewrite the tag in the manifest rather than `kubectl set image`, so the
# file in git always matches what the cluster is running.
sed -i '' "s|image: ${ECR}:.*|image: ${ECR}:${TAG}|" "$MANIFEST"

kubectl apply -f "$MANIFEST"
kubectl rollout status deployment/engine

echo "==> rolled out ${TAG}. The manifest changed; commit it:"
git --no-pager diff --stat -- "$MANIFEST"
