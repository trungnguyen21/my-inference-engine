# Infrastructure for running and testing the inference server on EC2.
#
# Scope is deliberately narrow: instances, a locked-down security group, and a
# key pair, in the default VPC. Terraform provisions machines; Kubernetes (in a
# later phase) schedules work onto them. Keeping those layers separate is the
# whole point of the split.
#
# Everything here is destroyable: `terraform destroy` is the cost control.

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

# The default VPC is used on purpose. A custom VPC would add a dozen resources
# that teach nothing about serving an inference engine.
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# Canonical's Ubuntu 24.04 for arm64.
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-arm64-server-*"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_key_pair" "engine" {
  key_name_prefix = "inference-engine-"
  public_key      = file(pathexpand(var.public_key_path))
}

resource "aws_security_group" "engine" {
  name_prefix = "inference-engine-"
  description = "Inference engine nodes: SSH, Kubernetes API, and NodePort, from one CIDR only"
  vpc_id      = data.aws_vpc.default.id

  tags = { Name = "inference-engine" }

  lifecycle {
    create_before_destroy = true
  }
}

# Rules are separate resources so they can change without replacing the group.
resource "aws_vpc_security_group_ingress_rule" "ssh" {
  security_group_id = aws_security_group.engine.id
  description       = "SSH"
  cidr_ipv4         = var.allowed_cidr
  from_port         = 22
  to_port           = 22
  ip_protocol       = "tcp"
}

resource "aws_vpc_security_group_ingress_rule" "kube_api" {
  security_group_id = aws_security_group.engine.id
  description       = "Kubernetes API, for kubectl from the laptop"
  cidr_ipv4         = var.allowed_cidr
  from_port         = 6443
  to_port           = 6443
  ip_protocol       = "tcp"
}

resource "aws_vpc_security_group_ingress_rule" "nodeport" {
  security_group_id = aws_security_group.engine.id
  description       = "NodePort range for the demo. A NodePort is used rather than an Ingress because proxies buffer SSE."
  cidr_ipv4         = var.allowed_cidr
  from_port         = 30000
  to_port           = 32767
  ip_protocol       = "tcp"
}

resource "aws_vpc_security_group_ingress_rule" "http_direct" {
  security_group_id = aws_security_group.engine.id
  description       = "Direct server port, for running it standalone or under Compose before Kubernetes exists"
  cidr_ipv4         = var.allowed_cidr
  from_port         = 8080
  to_port           = 8080
  ip_protocol       = "tcp"
}

# Nodes must reach each other on every port for k3s (flannel VXLAN, kubelet,
# etcd). Scoped to this security group, so it is not open to the VPC at large.
resource "aws_vpc_security_group_ingress_rule" "intra_node" {
  security_group_id            = aws_security_group.engine.id
  description                  = "All traffic between nodes in this group (k3s control plane, flannel, kubelet)"
  referenced_security_group_id = aws_security_group.engine.id
  ip_protocol                  = "-1"
}

resource "aws_vpc_security_group_egress_rule" "all" {
  security_group_id = aws_security_group.engine.id
  description       = "Outbound: package installs, the model download, container registries"
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
}

resource "aws_instance" "node" {
  count = var.node_count

  ami                         = data.aws_ami.ubuntu.id
  instance_type               = var.instance_type
  key_name                    = aws_key_pair.engine.key_name
  subnet_id                   = data.aws_subnets.default.ids[0]
  vpc_security_group_ids      = [aws_security_group.engine.id]
  associate_public_ip_address = true

  root_block_device {
    volume_size = var.root_volume_gb
    volume_type = "gp3"
    encrypted   = true
  }

  user_data = templatefile("${path.module}/bootstrap.sh.tftpl", {
    model_url   = var.model_url
    model_bytes = var.model_bytes
    go_version  = var.go_version
  })

  # Changing the bootstrap script should rebuild the machine, not leave a
  # half-provisioned one behind.
  user_data_replace_on_change = true

  tags = { Name = "inference-engine-${count.index + 1}" }
}
