output "node_ips" {
  description = "Public IPs, in node order. node-1 becomes the k3s server later."
  value       = aws_instance.node[*].public_ip
}

output "ssh" {
  description = "Ready-to-paste SSH commands."
  value       = [for ip in aws_instance.node[*].public_ip : "ssh ubuntu@${ip}"]
}

output "region" {
  value = var.region
}

output "instance_type" {
  value = var.instance_type
}

output "hourly_estimate_note" {
  description = "A reminder, not a billing API. Run `terraform destroy` when finished."
  value       = "${var.node_count} x ${var.instance_type} in ${var.region}. Run 'terraform destroy' when done; stopped-but-undestroyed EBS volumes still bill."
}
