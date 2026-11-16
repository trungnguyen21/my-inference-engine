# Deploy dev guide

How to get a change — code, config, or manifest — running on each of the three
deployment layers. The general shape is always the same:

```
edit → rebuild the image (if code changed) → redeploy → verify
```

The one invariant to re-check on every change: **`replicas x WORKERS x
THREADS` must stay at or under the cores actually available**, or throughput
falls off a cliff (measured: 7 s → 120 s).

## What kind of change is it?

| You changed | Compose | k3d | AWS (k3s on EC2) |
|---|---|---|---|
| Go/C++ code or Dockerfile | rebuild via `--build` | rebuild + re-import image | rebuild + push to ECR |
| env/config (`WORKERS`, `THREADS`, ...) | edit `compose.yaml`, `up -d` | edit manifest, `kubectl apply` | edit `eks/` manifest, `kubectl apply` |
| nginx config | edit `deploy/nginx.conf`, `up -d` | n/a (the Service replaces nginx) | n/a |
| k8s manifests | n/a | `kubectl apply -f deploy/k8s/` | `kubectl apply -f deploy/k8s/eks/` |
| nodes, security groups, model URL | n/a | n/a | `terraform apply` in `infra/` |

## Layer 1: Docker Compose

```sh
# code change — rebuild and restart in one step:
docker compose up --build -d

# config-only change (compose.yaml or deploy/nginx.conf):
docker compose up -d          # recreates only what changed; nginx reloads too

# verify:
curl -s localhost:8080/health | jq
docker compose ps             # both engine replicas healthy?
docker compose logs -f engine # watch the model load + OpenMP line
```

Scaling: `docker compose up -d --scale engine=3`. nginx re-resolves Docker DNS
at run time, so new replicas are picked up without touching the gateway — but
remember the thread budget before adding replicas.

## Layer 2: k3d

The image is side-loaded, not pulled (`imagePullPolicy: Never`), and the tag
never changes (`inference-engine:dev`). That has one sharp consequence:
**`kubectl apply` alone will not pick up a new image** — the pods must be
restarted so they start from the newly imported bytes.

```sh
# code change:
docker build -t inference-engine:dev .
k3d image import inference-engine:dev -c <cluster-name>
kubectl rollout restart deployment/engine
kubectl rollout status deployment/engine    # waits for the rolling update

# manifest change (env vars, replicas, probes):
kubectl apply -f deploy/k8s/
kubectl rollout status deployment/engine    # apply triggers the rollout itself

# verify:
kubectl get pods -o wide                    # Ready 1/1, spread over nodes
curl -s localhost:30080/health | jq         # via the k3d port mapping
kubectl logs deploy/engine --tail=20
```

The rolling update is `maxSurge: 1, maxUnavailable: 0`, so a bad image never
takes down the old pods — a pod that fails its readiness probe just blocks the
rollout. Diagnose with `kubectl describe pod <name>` (look at Events) and roll
back with `kubectl rollout undo deployment/engine`.

## Layer 3: AWS — k3s on Graviton EC2

Two independent halves. Know which one your change touches.

### Infrastructure (`infra/`, Terraform)

Owns: EC2 nodes, security groups, SSH keys, and the bootstrap script that
installs Docker and downloads the model to `/opt/models` on each node.

```sh
cd infra
terraform plan        # always read the plan first
terraform apply
terraform output      # node_ips, ssh command
```

Editing `bootstrap.sh.tftpl` **replaces the machines**
(`user_data_replace_on_change = true`) — that is deliberate, so nodes never
drift from their script, but it means the k3s cluster on them is rebuilt too.
Plan for that, don't be surprised by it.

### Workload (`deploy/k8s/eks/`, kubectl)

The image comes from ECR, so a code change is build → push → restart:

```sh
# build for arm64 (Graviton) and push:
aws ecr get-login-password --region ca-central-1 \
  | docker login --username AWS --password-stdin 327319899002.dkr.ecr.ca-central-1.amazonaws.com
docker build --platform linux/arm64 -t 327319899002.dkr.ecr.ca-central-1.amazonaws.com/inference-engine:dev .
docker push 327319899002.dkr.ecr.ca-central-1.amazonaws.com/inference-engine:dev

# same tag again, so a restart is what pulls the new digest:
kubectl rollout restart deployment/engine
kubectl rollout status deployment/engine

# manifest change:
kubectl apply -f deploy/k8s/eks/
```

Differences from the k3d manifests, so you edit the right thing:

- `replicas: 4` with `THREADS: 2` — sized for the 2x c7g.xlarge budget
  (8 vCPUs total). Change either side of that product, re-do the math.
- The model reaches nodes twice over: the Terraform bootstrap downloads it,
  and the `model-download` DaemonSet does the same thing idempotently for
  nodes added later. A new node needs neither hand-holding — the DaemonSet
  lands there unprompted, and the engine pod's `wait-for-model`
  initContainer blocks until the file exists.
- `hostPath` is strict (`type: Directory`): if the mount fails, the fix is
  the DaemonSet or bootstrap, never `mkdir` by hand on the node.

### Verify on AWS

```sh
kubectl get pods -o wide                     # pods spread across both nodes?
curl -s http://<node_ip>:30080/health | jq   # any node works; NodePort spans the cluster
```

## Pitfalls that have actually bitten

- **Forgot `k3d image import`** — pods restart onto the *old* image with no
  error anywhere, because the tag resolved locally. If a fix "didn't take",
  check this first.
- **Thread oversubscription** — raising `replicas` without lowering `THREADS`
  (or vice versa). The server logs a warning line at startup; read it.
- **Ingress/proxies that buffer** — anything inserted in front of the server
  must not buffer responses, or SSE arrives as one burst. This is why nginx
  has `proxy_buffering off` and the cluster uses NodePort, not Ingress.
- **Killing a pod mid-generation** — the server drains for 60 s on SIGTERM
  and the pod spec grants 65 (`terminationGracePeriodSeconds`). If you
  shorten one, shorten both.
