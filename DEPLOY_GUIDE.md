# Deploy guide — making changes yourself

Three layers, three workflows. The first question for any change is "which
layer does this live in?" — everything else follows from the answer.

| Layer | Lives in | Changed by | Examples |
|---|---|---|---|
| Image (code) | `server/`, `src/`, `Dockerfile`, **`server/demo.html`** | build → push → rolling update | API endpoints, engine code, the dashboard |
| Manifests | `deploy/k8s/` | `kubectl apply` | replicas, THREADS, probes, which model |
| Infrastructure | `infra/eks/` | `terraform apply` | instance type, node count, allowed IP |

Terraform never redeploys code, and kubectl never resizes machines.

---

## 1. Code changes (including the demo dashboard)

`demo.html` is baked into the image at build time — editing the file changes
nothing running until a new image rolls out. Same for all Go and C++ code.

**The short way** — one command does all of the below with an automatic tag
(the git commit SHA, plus a timestamp suffix if the tree is dirty):

```bash
./deploy/release.sh
# then commit the manifest change it made
```

**The long way**, which is what the script does:

```bash
# 1. Pick a NEW tag. Never reuse one: with imagePullPolicy IfNotPresent,
#    re-pushing the same tag deploys nothing — nodes already "have" it.
TAG=v3   # or: TAG=$(git rev-parse --short HEAD)
ECR=327319899002.dkr.ecr.ca-central-1.amazonaws.com/inference-engine

# 2. Build (the Mac is arm64, same as Graviton — no cross-build) and push.
aws ecr get-login-password --region ca-central-1 \
  | docker login --username AWS --password-stdin $ECR
docker build -t $ECR:$TAG .
docker push $ECR:$TAG

# 3. Update the tag in deploy/k8s/eks/deployment.yaml (keeps git honest),
#    then apply. Editing the file + apply is the same as `kubectl set image`,
#    but the repo still matches reality afterwards.
kubectl apply -f deploy/k8s/eks/deployment.yaml
kubectl rollout status deployment/engine

# Broken? One command back:
kubectl rollout undo deployment/engine
```

The rollout is zero-downtime by construction: maxSurge 1 / maxUnavailable 0
means a new pod must pass its readiness probe before an old one is stopped.

## 2. Manifest changes

Edit the YAML, apply, watch. That's the whole workflow.

```bash
kubectl apply -f deploy/k8s/eks/
kubectl rollout status deployment/engine
```

Only pod-*template* changes (env, image, probes) trigger a rolling restart.
`replicas` just adds or removes pods; a Service/DaemonSet edit applies to
those objects alone.

**Capacity rule** when touching `replicas` or `THREADS`: on 2x c7g.xlarge
the budget is `replicas x THREADS <= 8 vCPUs`.

| Layout | Concurrent streams | Per-stream speed |
|---|---|---|
| 2 x THREADS=4 | 2 | fastest |
| 4 x THREADS=2 | 4 | ~60–70% |
| 8 x THREADS=1 | 8 (one per dashboard panel) | slowest |

Requests beyond the concurrent capacity are rejected busy, not queued —
red panes on the dashboard mean more panels than workers.

**Changing the model**: new URL + byte count in
`deploy/k8s/eks/model-download.yaml`, new filename in the Deployment's
`MODEL` env, apply both. The DaemonSet refreshes each node, the env change
rolls the pods.

**Known wrinkle**: after a rolling update the 4+4 node spread can drift
(the constraint only applies at scheduling time, and terminating pods still
counted). Fix by deleting one pod on the crowded node — its replacement is
forced onto the emptier one:

```bash
kubectl get pods -o wide            # see the imbalance
kubectl delete pod <one-on-the-crowded-node>
```

## 3. Infrastructure changes

```bash
cd infra/eks
# edit main.tf / variables.tf / terraform.tfvars
terraform plan -out=tfplan          # ALWAYS read the plan first
terraform apply tfplan
```

The plan is the safety net: "1 to add, 0 to destroy" reads very differently
from "2 to destroy". If a change *replaces* the node group, expect ~5 min of
reduced capacity while nodes cycle.

Home IP changed (kubectl suddenly times out)? Update `allowed_cidr` in
`terraform.tfvars`, plan, apply — ~30 s.

---

## Rehearsing locally first (free)

The k3d cluster is disposable; recreate it any time:

```bash
k3d cluster create engine --servers 1 --agents 2 \
  --k3s-arg "--node-taint=CriticalAddonsOnly=true:NoExecute@server:0" \
  --volume "$(pwd)/models:/models@all" -p "30080:30080@agent:0"
docker build -t inference-engine:dev .
k3d image import inference-engine:dev -c engine
kubectl apply -f deploy/k8s/
```

Local code loop: rebuild → `k3d image import` → `kubectl rollout restart
deployment/engine`. (The import replaces the sideloaded image in place, so
restart suffices — no tag discipline needed locally.)

Switch kubectl between clusters:

```bash
kubectl config get-contexts     # k3d-engine vs the EKS arn
kubectl config use-context <name>
```

k9s follows whichever context is active.

---

## Spin up from zero / tear down

Up (~25 min, ~$0.45/hr while running):

```bash
cd infra/eks
echo "allowed_cidr = \"$(curl -4 -s ifconfig.me)/32\"" > terraform.tfvars
terraform init && terraform plan -out=tfplan && terraform apply tfplan
aws eks update-kubeconfig --region ca-central-1 --name inference-engine
# push the image (workflow 1), then:
kubectl apply -f ../../deploy/k8s/eks/
# metrics are an add-on, not part of the repo manifests — re-install per
# cluster (see "Watching resource usage"):
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl get nodes -o wide            # EXTERNAL-IP column
open http://<node-external-ip>:30080/demo
```

Down — **the step that stops billing**:

```bash
cd infra/eks && terraform destroy
# then verify: EC2 console shows no running instances
```

Destroy removes the ECR images too (`force_delete`), so the next spin-up
re-pushes the image. Nothing of value lives in the cluster; the repo is the
source of truth.

---

## Watching resource usage (CPU / RAM per pod and node)

Needs **metrics-server**, a cluster add-on. k3s/k3d ship it built-in; on EKS
install it once per cluster (it does not survive a destroy/respin, because it
was applied to the cluster, not committed to `deploy/k8s/`):

```bash
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
# first numbers appear after a scrape cycle, ~30 s
```

Then:

```bash
kubectl top nodes                   # per-node CPU/MEM — imbalance shows here
kubectl top pods --sort-by=cpu      # busiest pods first; ~1 core per active
                                    #   stream at THREADS=1
watch kubectl top nodes             # live view during loadgen
```

k9s picks the metrics up automatically (CPU/MEM columns in the pod view;
`%CPU/R` is percent of the pod's *request*, which reads >100% here because
requests are deliberately small — expected, not an alarm).

Reading the numbers:

- Idle engine pods report ~55 Mi despite a 469 MiB model: the mmap'd weights
  are counted once per *node* (shared page cache), not per pod. The per-node
  memory figure carries it.
- Imbalance demo: `kubectl top nodes` before and after fixing a 5/3 pod
  split (delete one pod on the hot node) shows one node hot / one cool, then
  even.
- metrics-server keeps only the latest scrape — live gauges, no history.
  Graphs over time would be Prometheus + Grafana, a deliberately separate
  phase.

## When something is wrong

Ask the cluster why it hasn't converged, in this order:

```bash
kubectl get pods -o wide        # what state is everything in?
kubectl describe pod <name>     # events: probe failures, scheduling, pulls
kubectl logs <name>             # what the app itself said
kubectl logs <name> -c wait-for-model   # if stuck in Init:0/1
```

Frequent states and what they mean:

- `Init:0/1` — waiting for the model download; check the model-download pod.
- `ImagePullBackOff` — tag doesn't exist in ECR (forgot to push?).
- `Pending` — no node fits; describe shows which constraint failed.
- `CrashLoopBackOff` — the engine itself is failing; read its logs.
- Dashboard pane red with `HTTP 503` — more panels than workers; see the
  capacity table.
- Old dashboard after a deploy — reused image tag (see workflow 1).
