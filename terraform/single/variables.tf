variable "key_name" {
  description = "Name of the EC2 key pair (must already exist in the AWS account). In Learner Lab us-east-1 use 'vockey'."
  type        = string
}

variable "region" {
  description = "AWS region. Learner Lab supports us-east-1 (default, vockey available) and us-west-2."
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type. Must fit ALL N worker containers + registry + OS/Docker overhead. t3.large (8 GB) handles up to 8 workers with the full dataset; t3.medium (4 GB) is borderline for 3 workers. Learner Lab does NOT support t3.xlarge or larger."
  type        = string
  default     = "t3.large"
}

variable "volume_size" {
  description = "EBS root volume size (GB). All worker containers and dataset partitions share this disk. Recommended range: 20 GB (up to 5 workers) to 30 GB (8 workers, full dataset)."
  type        = number
  default     = 20
}

variable "availability_zone" {
  description = "AZ to place the instance in. Consistent with the region (e.g. us-east-1a for us-east-1, us-west-2a for us-west-2). All traffic is internal to the Docker bridge network so cross-AZ cost does not apply in single-EC2 mode, but keeping this consistent with multi-instance mode avoids confusion."
  type        = string
  default     = "us-east-1a"
}
