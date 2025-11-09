# Robo Dunk

Goal: Train a neural network to successfully get the ball into the basket, using reinforcement learning.

# Development Roadmap

1. Project Setup: Repo/git setup, packages, githooks (linting/formatting)
2. Tests: Unit tests for a testing-led development approach
3. Environment: A cannon shoots a ball at a robot, which can then hit it into the basket
4. Train: Use colab to train a PPO model
5. Experimenting: Track with tensorboard
6. Frontend: Build a frontend with Streamlit
7. Containerise: Using Docker and building a FastAPI backend. Deploy on AWS
8. Monitoring & CI/CD

## Monitoring: Prometheus Monitoring & Grafana Dashboard

Here are the relevant commands for local setup:

Run

`docker-compose up -d`

This orchestrates both services. Then start the frontend app

`streamlit run sandbox.py`

And then go to:

http://localhost:9090/

http://localhost:3000/
