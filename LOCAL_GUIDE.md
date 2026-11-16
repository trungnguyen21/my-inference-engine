# Build (from scratch)

cd "/Users/trungnguyen/Desktop/Coding projects/my-own-inference-engine"

1. Build the engine image (if not already built)

`docker build -t inference-engine:dev .`

2. Create the 2-node cluster: 1 server + 1 agent,
   models/ mounted into every node, NodePort 30080 exposed to localhost

```sh
k3d cluster create engine --servers 1 --agents 1 
\--volume "$(pwd)/models:/models@all" -p "30080:30080@server:0"
```

3. Sideload the image into the cluster (no registry in k3d)
`k3d image import inference-engine:dev -c engine`

4. Deploy
`kubectl apply -f deploy/k8s/`

5. Wait until both pods are ready
`kubectl rollout status deployment/engine`

6. Verify
```sh
kubectl get pods -o wide
curl -s http://localhost:30080/health/ready
```

Then: dashboard at http://localhost:30080/demo, loadgen with cd demo && go run ./loadgen -url http://localhost:30080 -users 4.

After rebuilding the image (code changes), repeat steps 1 and 3, then kubectl rollout restart deployment/engine.

# Teardown

Two levels, depending on what you want:

### Soft: remove just the workload, keep the cluster running
`kubectl delete -f deploy/k8s/`

### Full: delete the whole cluster (all three Docker containers, kubeconfig entry cleaned up)
`k3d cluster delete engine`

k3d cluster delete engine is the complete teardown — nothing keeps running, and nothing costs anything afterward. Your models/ folder and the inference-engine:dev image in local Docker are untouched, so a rebuild is fast (the slow parts — model download, image build — don't repeat).

## Pause instead of teardown

If you just want to free up CPU/RAM without losing the cluster:

```sh
k3d cluster stop engine    # stops the node containers
k3d cluster start engine   # brings everything back, pods restart on their own
```