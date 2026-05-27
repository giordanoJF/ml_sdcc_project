output "registry_public_ip" {
  description = "Public IP of the registry instance (used for SSH and health checks)"
  value       = aws_instance.registry.public_ip
}

output "registry_private_ip" {
  description = "Private IP of the registry instance (used by workers as REGISTRY_URL)"
  value       = aws_instance.registry.private_ip
}

output "worker_public_ips" {
  description = "Public IPs of worker instances indexed by worker_id (used for SSH)"
  value       = aws_instance.worker[*].public_ip
}

output "worker_private_ips" {
  description = "Private IPs of worker instances indexed by worker_id (used as MY_HOST for gRPC registration)"
  value       = aws_instance.worker[*].private_ip
}
