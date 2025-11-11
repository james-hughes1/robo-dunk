terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "instance_name" {
  description = "Name for the Lightsail instance"
  type        = string
  default     = "rl-monitoring-app"
}

variable "bundle_id" {
  description = "Lightsail bundle ID (instance size)"
  type        = string
  default     = "medium_3_0" # 2 vCPU, 4GB RAM, 60GB SSD - $20/month
  # Options:
  # micro_3_0:  1 vCPU, 512MB RAM, 20GB SSD - $3.50/month (too small)
  # small_3_0:  1 vCPU, 2GB RAM, 40GB SSD - $10/month (might work)
  # medium_3_0: 2 vCPU, 4GB RAM, 60GB SSD - $20/month (recommended)
  # large_3_0:  2 vCPU, 8GB RAM, 80GB SSD - $40/month
}

variable "blueprint_id" {
  description = "Lightsail OS blueprint"
  type        = string
  default     = "ubuntu_22_04"
}

variable "ssh_key_name" {
  description = "Name for SSH key pair"
  type        = string
  default     = "rl-app-key"
}

variable "availability_zone" {
  description = "Availability zone"
  type        = string
  default     = "us-east-1a"
}

# SSH Key Pair (optional - Lightsail can generate one for you)
resource "aws_lightsail_key_pair" "main" {
  name = var.ssh_key_name
}

# Lightsail Instance
resource "aws_lightsail_instance" "app" {
  name              = var.instance_name
  availability_zone = var.availability_zone
  blueprint_id      = var.blueprint_id
  bundle_id         = var.bundle_id
  key_pair_name     = aws_lightsail_key_pair.main.name

  user_data = file("${path.module}/user_data.sh")

  tags = {
    Environment = "production"
    Project     = "rl-monitoring"
  }
}

# Static IP
resource "aws_lightsail_static_ip" "app" {
  name = "${var.instance_name}-ip"
}

# Attach Static IP to Instance
resource "aws_lightsail_static_ip_attachment" "app" {
  static_ip_name = aws_lightsail_static_ip.app.name
  instance_name  = aws_lightsail_instance.app.name
}

# Open ports for services
resource "aws_lightsail_instance_public_ports" "app" {
  instance_name = aws_lightsail_instance.app.name

  # HTTP
  port_info {
    protocol  = "tcp"
    from_port = 80
    to_port   = 80
    cidrs     = ["0.0.0.0/0"]
  }

  # HTTPS
  port_info {
    protocol  = "tcp"
    from_port = 443
    to_port   = 443
    cidrs     = ["0.0.0.0/0"]
  }

  # Streamlit
  port_info {
    protocol  = "tcp"
    from_port = 8501
    to_port   = 8501
    cidrs     = ["0.0.0.0/0"]
  }

  # Grafana
  port_info {
    protocol  = "tcp"
    from_port = 3000
    to_port   = 3000
    cidrs     = ["0.0.0.0/0"]
  }

  # Prometheus (optional - can restrict)
  port_info {
    protocol  = "tcp"
    from_port = 9090
    to_port   = 9090
    cidrs     = ["0.0.0.0/0"]
  }

  # SSH
  port_info {
    protocol  = "tcp"
    from_port = 22
    to_port   = 22
    cidrs     = ["0.0.0.0/0"]
  }
}

# Outputs
output "instance_public_ip" {
  description = "Public IP address of the instance"
  value       = aws_lightsail_static_ip.app.ip_address
}

output "instance_name" {
  description = "Name of the Lightsail instance"
  value       = aws_lightsail_instance.app.name
}

output "ssh_command" {
  description = "SSH command to connect to instance"
  value       = "ssh -i ${var.ssh_key_name}.pem ubuntu@${aws_lightsail_static_ip.app.ip_address}"
}

output "streamlit_url" {
  description = "Streamlit app URL"
  value       = "http://${aws_lightsail_static_ip.app.ip_address}:8501"
}

output "grafana_url" {
  description = "Grafana dashboard URL"
  value       = "http://${aws_lightsail_static_ip.app.ip_address}:3000"
}

output "ssh_private_key" {
  description = "SSH private key (save this!)"
  value       = aws_lightsail_key_pair.main.private_key
  sensitive   = true
}
