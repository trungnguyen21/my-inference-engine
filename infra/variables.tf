variable "region" {
  description = "AWS region. ca-central-1 keeps latency low from Toronto."
  type        = string
  default     = "ca-central-1"
}

variable "node_count" {
  description = <<-EOT
    How many instances to run.

    1 is enough for Phase 5-6 (demo kit, single server, Compose).
    Raise to 2 for the Phase 7-8 k3s cluster: node-1 becomes the k3s server,
    the rest join as agents.
  EOT
  type        = number
  default     = 1
}

variable "instance_type" {
  description = <<-EOT
    Graviton (arm64) on purpose: the engine has NEON kernels, so arm64 uses the
    optimized path rather than the scalar fallback.

    Avoid t-class burstable types. CPU credits throttle sustained load, which
    is exactly what inference is.
  EOT
  type        = string
  default     = "c7g.large" # 2 vCPU, 4 GiB. c7g.xlarge (4 vCPU, 8 GiB) for the demo.
}

variable "allowed_cidr" {
  description = <<-EOT
    The only CIDR allowed to reach SSH, the Kubernetes API, and the NodePort.

    Defaults to nothing on purpose: `terraform apply` will fail rather than
    quietly exposing a Kubernetes API server to the internet. Set it in
    terraform.tfvars, normally to "<your-ip>/32".
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

variable "public_key_path" {
  description = "Local SSH public key to install on the instances."
  type        = string
  default     = "~/.ssh/id_ed25519.pub"
}

variable "model_url" {
  description = <<-EOT
    GGUF model, downloaded on the instance rather than uploaded.

    This matters: the file is ~469 MiB and a home upload link makes that a
    slow, repeated annoyance. AWS pulls it in seconds.
  EOT
  type        = string
  default     = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
}

variable "model_bytes" {
  description = "Expected size of the model, checked after download so a truncated file fails loudly."
  type        = number
  default     = 491400032
}

variable "go_version" {
  description = "Ubuntu 24.04 ships Go 1.22; server/go.mod needs newer, so install the official tarball."
  type        = string
  default     = "1.26.3"
}

variable "root_volume_gb" {
  description = "Room for the model, the llama.cpp build tree, and container images."
  type        = number
  default     = 30
}
