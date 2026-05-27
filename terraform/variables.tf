variable "num_workers" {
  description = "Number of worker EC2 instances (must match network.num_workers in config.yaml)"
  type        = number
}

variable "instance_type_worker" {
  description = "EC2 instance type for worker nodes. t3.small (2 GB) is sufficient for all worker counts with the full FEMNIST dataset."
  type        = string
  default     = "t3.small"
}

variable "instance_type_registry" {
  description = "EC2 instance type for the discovery registry. t3.micro (1 GB) is more than enough for the Flask server."
  type        = string
  default     = "t3.micro"
}

variable "volume_size_worker" {
  description = "EBS root volume size (GB) for each worker instance. Recommended range: 15 (8 workers, --sf 0.05) to 30 (3 workers, full dataset). Default 20 covers all cases."
  type        = number
  default     = 20
}

variable "volume_size_registry" {
  description = "EBS root volume size (GB) for the registry instance. 8 GB is sufficient; no reason to exceed 15 GB."
  type        = number
  default     = 8
}

variable "availability_zone" {
  description = "AZ to pin ALL instances to. Same-AZ private traffic is free; cross-AZ costs $0.01/GB each direction. Must be a valid AZ for the configured region (e.g. us-east-1a for us-east-1, us-west-2a for us-west-2)."
  type        = string
  default     = "us-east-1a"
}

variable "key_name" {
  description = "Name of the EC2 key pair (must already exist in the AWS account). In Learner Lab us-east-1 use 'vockey'."
  type        = string
}

variable "region" {
  description = "AWS region. Learner Lab supports us-east-1 (default, vockey available) and us-west-2."
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
