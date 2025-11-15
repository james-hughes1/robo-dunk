#!/bin/bash
set -e

echo "Pulling latest image..."
docker pull $ECR_REPO_URL:latest

echo "Stopping old container if exists..."
docker stop streamlit || true
docker rm streamlit || true

echo "Starting new container..."
docker run -d \
  --name streamlit \
  -p 8501:8501 \
  -e STREAMLIT_EMAIL_OPT_IN=false \
  $ECR_REPO_URL:latest
