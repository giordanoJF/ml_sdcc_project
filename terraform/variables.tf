variable "num_workers" {
  description = "Number of worker EC2 instances (must match network.num_workers in config.yaml)"
  type        = number
}

variable "instance_type_worker" {
  description = "EC2 instance type for worker nodes (e.g. t3.small, t3.medium)"
  type        = string
  default     = "t3.small"
}

variable "instance_type_registry" {
  description = "EC2 instance type for the discovery registry"
  type        = string
  default     = "t3.micro"
}

variable "key_name" {
  description = "Name of the EC2 key pair (must already exist in the AWS account)"
  type        = string
}

variable "region" {
  description = "AWS region (Learner Lab supports us-east-1 and us-west-2; vockey key pair available in us-east-1 only)"
  type        = string
  default     = "us-east-1"
}

variable "registry_port" {
  description = "TCP port for the discovery registry HTTP server"
  type        = number
  default     = 5000
}

variable "grpc_port" {
  description = "TCP port for the worker gRPC server"
  type        = number
  default     = 50051
}
