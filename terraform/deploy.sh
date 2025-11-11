#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== RL Monitoring App Deployment Script ===${NC}"

# Get the server IP from Terraform output
SERVER_IP=$(terraform output -raw instance_public_ip 2>/dev/null || echo "")

if [ -z "$SERVER_IP" ]; then
    echo -e "${RED}Error: Could not get server IP from Terraform${NC}"
    echo "Run 'terraform apply' first"
    exit 1
fi

echo -e "${GREEN}Deploying to: ${SERVER_IP}${NC}"

# Save SSH key
echo -e "${YELLOW}Saving SSH private key...${NC}"
terraform output -raw ssh_private_key > rl-app-key.pem
chmod 600 rl-app-key.pem

# Wait for instance to be ready
echo -e "${YELLOW}Waiting for instance to be ready...${NC}"
sleep 30

# Test SSH connection
echo -e "${YELLOW}Testing SSH connection...${NC}"
for i in {1..10}; do
    if ssh -i rl-app-key.pem -o StrictHostKeyChecking=no -o ConnectTimeout=5 ubuntu@${SERVER_IP} "echo 'Connected'" 2>/dev/null; then
        echo -e "${GREEN}SSH connection successful!${NC}"
        break
    fi
    echo "Waiting for SSH... attempt $i/10"
    sleep 10
done

# Create remote directory
echo -e "${YELLOW}Creating app directory on server...${NC}"
ssh -i rl-app-key.pem ubuntu@${SERVER_IP} "mkdir -p ~/app"

# Copy project files (excluding unnecessary files)
echo -e "${YELLOW}Copying project files...${NC}"
rsync -avz --progress \
    -e "ssh -i rl-app-key.pem -o StrictHostKeyChecking=no" \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.env' \
    --exclude 'terraform.tfstate*' \
    --exclude '*.pem' \
    --exclude '.terraform' \
    ./ ubuntu@${SERVER_IP}:~/app/

# Build and start containers
echo -e "${YELLOW}Building and starting Docker containers...${NC}"
ssh -i rl-app-key.pem ubuntu@${SERVER_IP} << 'ENDSSH'
cd ~/app

# Stop existing containers if any
docker compose down 2>/dev/null || true

# Build images
docker compose build --no-cache

# Start services
docker compose up -d

# Wait for services to start
echo "Waiting for services to start..."
sleep 15

# Check status
docker compose ps

echo "=== Deployment complete! ==="
ENDSSH

# Display access URLs
echo -e "${GREEN}"
echo "=================================="
echo "Deployment Successful!"
echo "=================================="
echo -e "${NC}"
echo -e "Streamlit App: ${GREEN}http://${SERVER_IP}:8501${NC}"
echo -e "Grafana:       ${GREEN}http://${SERVER_IP}:3000${NC} (admin/admin)"
echo -e "Prometheus:    ${GREEN}http://${SERVER_IP}:9090${NC}"
echo ""
echo -e "SSH Access:    ${YELLOW}ssh -i rl-app-key.pem ubuntu@${SERVER_IP}${NC}"
echo ""
echo -e "${YELLOW}Note: It may take 1-2 minutes for services to fully start${NC}"
