output "public_ip" {
  description = "Public IP of the instance (used for SSH and SCP)"
  value       = aws_instance.single.public_ip
}

output "private_ip" {
  description = "Private IP of the instance"
  value       = aws_instance.single.private_ip
}
