terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

# Latest Ubuntu 22.04 LTS AMI (Canonical)
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_security_group" "fl" {
  name        = "fl-sdcc"
  description = "P2P Federated Learning - SDCC project"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "Registry HTTP"
    from_port   = var.registry_port
    to_port     = var.registry_port
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "gRPC worker"
    from_port   = var.grpc_port
    to_port     = var.grpc_port
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # All traffic between instances in the same security group
  ingress {
    description = "Intra-cluster"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    self        = true
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Project = "fl-sdcc" }
}

# Installs Docker at first boot via user_data (runs in background while
# Terraform reports the apply as complete; aws_deploy.py waits for Docker
# to be ready before proceeding with the build step).
locals {
  user_data = <<-EOF
    #!/bin/bash
    apt-get update -y
    apt-get install -y docker.io
    systemctl enable docker
    systemctl start docker
    usermod -aG docker ubuntu
  EOF
}

resource "aws_instance" "registry" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type_registry
  key_name               = var.key_name
  vpc_security_group_ids = [aws_security_group.fl.id]
  availability_zone      = var.availability_zone
  user_data              = local.user_data


  root_block_device {
    volume_type = "gp3"
    volume_size = var.volume_size_registry
  }

  tags = {
    Name    = "fl-registry"
    Project = "fl-sdcc"
    Role    = "registry"
  }
}

resource "aws_instance" "worker" {
  count                  = var.num_workers
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type_worker
  key_name               = var.key_name
  vpc_security_group_ids = [aws_security_group.fl.id]
  availability_zone      = var.availability_zone
  user_data              = local.user_data


  root_block_device {
    volume_type = "gp3"
    volume_size = var.volume_size_worker
  }

  tags = {
    Name     = "fl-worker-${count.index}"
    Project  = "fl-sdcc"
    Role     = "worker"
    WorkerID = tostring(count.index)
  }
}
