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

resource "aws_security_group" "fl_single" {
  name        = "fl-sdcc-single"
  description = "P2P Federated Learning - single EC2 mode"

  # SSH access for project upload and management
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # All worker-to-registry and worker-to-worker gRPC traffic uses Docker's
  # internal bridge network — no extra ingress rules needed.
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
# to be ready before printing the "instance ready" message).
locals {
  user_data = <<-EOF
    #!/bin/bash
    apt-get update -y
    apt-get install -y docker.io docker-compose-v2
    systemctl enable docker
    systemctl start docker
    usermod -aG docker ubuntu
  EOF
}

resource "aws_instance" "single" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  key_name               = var.key_name
  vpc_security_group_ids = [aws_security_group.fl_single.id]
  availability_zone      = var.availability_zone
  user_data              = local.user_data


  root_block_device {
    volume_type = "gp3"
    volume_size = var.volume_size
  }

  tags = {
    Name    = "fl-single"
    Project = "fl-sdcc"
  }
}
