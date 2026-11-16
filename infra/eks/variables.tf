variable "region" {
  description = "AWS region. ca-central-1 keeps latency low from Toronto."
  type        = string
  default     = "ca-central-1"
}

variable "cluster_name" {
  description = "EKS cluster name; also what `aws eks update-kubeconfig` takes."
  type        = string
  default     = "inference-engine"
}

variable "node_count" {
  description = <<-EOT
    Worker instances. Unlike the k3s layout there is no server among them —
    AWS runs the control plane — so both nodes are pure inference capacity.
  EOT
  type        = number
  default     = 2
}

variable "instance_type" {
  description = <<-EOT
    Graviton (arm64) for the NEON kernels, non-burstable for sustained load —
    same reasoning as ../variables.tf.

    c7g.2xlarge = 8 vCPU, 16 GiB per node. The capacity budget is
    (parallel generations x THREADS) <= total vCPUs: with 2 nodes that is
    16 cores, so 4 pods x THREADS=4 serves 4 full-speed concurrent streams.
  EOT
  type        = string
  default     = "c7g.2xlarge"
}

variable "allowed_cidr" {
  description = <<-EOT
    The only CIDR allowed to reach the Kubernetes API and the NodePort.
    No default, same refusal as ../variables.tf: set "<your-ip>/32" in
    terraform.tfvars rather than exposing the API server to the internet.
  EOT
  type        = string

  validation {
    condition     = can(cidrnetmask(var.allowed_cidr))
    error_message = "allowed_cidr must be valid CIDR, e.g. 203.0.113.4/32."
  }
  validation {
    condition     = var.allowed_cidr != "0.0.0.0/0"
    error_message = "Refusing 0.0.0.0/0: that exposes the Kubernetes API to the internet."
  }
}

variable "root_volume_gb" {
  description = "Per-node disk: the pulled image, the model download, logs."
  type        = number
  default     = 30
}
