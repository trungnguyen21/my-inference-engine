# EKS variant of the cluster: AWS runs the control plane as a managed service
# (that is the $0.10/hr), and the only machines we own are pure workers. The
# k3s "server node doing double duty" question disappears — every instance
# here earns its keep serving inference, and the brain cannot be killed.
#
# Separate root module from ../ on purpose: the EC2/k3s infra and this cluster
# have different lifecycles and different state files. `terraform destroy`
# here removes everything, including the registry and its images.

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project   = "my-own-inference-engine"
      ManagedBy = "terraform"
    }
  }
}

# Same stance as ../main.tf: the default VPC teaches nothing less. EKS needs
# subnets in at least two availability zones; the default VPC has one per AZ.
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# ---------------------------------------------------------------------------
# IAM. EKS will not act on your behalf implicitly; both the control plane and
# the nodes need roles stating exactly what they may do. These two roles with
# AWS-managed policies are the floor, not a simplification.
# ---------------------------------------------------------------------------

resource "aws_iam_role" "cluster" {
  name_prefix = "inference-eks-cluster-"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "eks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "cluster" {
  role       = aws_iam_role.cluster.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
}

resource "aws_iam_role" "node" {
  name_prefix = "inference-eks-node-"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

# WorkerNode: join the cluster. CNI: manage pod networking. ECR read-only:
# pull the engine image from the registry below.
resource "aws_iam_role_policy_attachment" "node_worker" {
  role       = aws_iam_role.node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
}

resource "aws_iam_role_policy_attachment" "node_cni" {
  role       = aws_iam_role.node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
}

resource "aws_iam_role_policy_attachment" "node_ecr" {
  role       = aws_iam_role.node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

# ---------------------------------------------------------------------------
# The cluster: just the control plane. Nodes are a separate resource.
# ---------------------------------------------------------------------------

resource "aws_eks_cluster" "engine" {
  name     = var.cluster_name
  role_arn = aws_iam_role.cluster.arn

  vpc_config {
    subnet_ids = data.aws_subnets.default.ids

    # The API server gets a public endpoint (so kubectl/k9s work from the
    # laptop without a VPN) but, exactly like ../variables.tf refuses
    # 0.0.0.0/0 for port 6443, it only answers your CIDR.
    endpoint_public_access  = true
    endpoint_private_access = true
    public_access_cidrs     = [var.allowed_cidr]
  }

  depends_on = [aws_iam_role_policy_attachment.cluster]
}

# The managed node group: EKS's ASG-backed equivalent of ../main.tf's
# aws_instance count. Graviton for the same reason as before — the engine's
# NEON kernels — hence the ARM AMI type.
resource "aws_eks_node_group" "workers" {
  cluster_name    = aws_eks_cluster.engine.name
  node_group_name = "inference-workers"
  node_role_arn   = aws_iam_role.node.arn
  subnet_ids      = data.aws_subnets.default.ids

  instance_types = [var.instance_type]
  ami_type       = "AL2023_ARM_64_STANDARD"
  disk_size      = var.root_volume_gb

  scaling_config {
    desired_size = var.node_count
    min_size     = var.node_count
    max_size     = var.node_count + 1 # headroom for a live "add a node" demo
  }

  depends_on = [
    aws_iam_role_policy_attachment.node_worker,
    aws_iam_role_policy_attachment.node_cni,
    aws_iam_role_policy_attachment.node_ecr,
  ]
}

# EKS puts every node in a cluster security group it creates itself. The
# NodePort range is not open in it by default; this rule is what lets the
# demo dashboard and loadgen reach the Service from the laptop.
resource "aws_vpc_security_group_ingress_rule" "nodeport" {
  security_group_id = aws_eks_cluster.engine.vpc_config[0].cluster_security_group_id
  description       = "NodePort range for the demo, from one CIDR only"
  cidr_ipv4         = var.allowed_cidr
  from_port         = 30000
  to_port           = 32767
  ip_protocol       = "tcp"
}

# ---------------------------------------------------------------------------
# Registry. k3d sideloads images; EKS nodes pull them. ECR is the missing
# piece: build arm64 locally (the Mac is arm64, so no cross-build), push once,
# every node pulls it. The push commands are in the outputs.
# ---------------------------------------------------------------------------

resource "aws_ecr_repository" "engine" {
  name = "inference-engine"

  # Let `terraform destroy` delete the repo even with images in it; the
  # registry is as disposable as the cluster.
  force_delete = true
}
