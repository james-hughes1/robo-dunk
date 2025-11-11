#!/bin/bash
set -e

# Log everything
exec > >(tee /var/log/user-data.log)
exec 2>&1

echo "=== Starting instance setup at $(date) ==="

# Update system
apt-get update
apt-get upgrade -y

# Install Docker
echo "=== Installing Docker ==="
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker ubuntu

# Install Docker Compose
echo "=== Installing Docker Compose ==="
apt-get install -y docker-compose-plugin

# Install other useful tools
apt-get install -y git curl wget vim htop

# Create app directory
mkdir -p /home/ubuntu/app
chown ubuntu:ubuntu /home/ubuntu/app

# Enable Docker service
systemctl enable docker
systemctl start docker

# Set up log rotation for Docker
cat > /etc/docker/daemon.json <<EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
EOF

systemctl restart docker

echo "=== Setup complete at $(date) ==="
echo "=== Ready to deploy application ==="
