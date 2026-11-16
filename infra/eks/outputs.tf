output "kubeconfig_command" {
  description = "Run once after apply; kubectl and k9s then point at the cluster."
  value       = "aws eks update-kubeconfig --region ${var.region} --name ${aws_eks_cluster.engine.name}"
}

output "ecr_repository_url" {
  description = "Where the engine image lives. Nodes pull from here."
  value       = aws_ecr_repository.engine.repository_url
}

output "image_push_commands" {
  description = "Build locally (the Mac is already arm64), push once."
  value       = <<-EOT
    aws ecr get-login-password --region ${var.region} | docker login --username AWS --password-stdin ${aws_ecr_repository.engine.repository_url}
    docker build -t ${aws_ecr_repository.engine.repository_url}:dev .
    docker push ${aws_ecr_repository.engine.repository_url}:dev
  EOT
}
