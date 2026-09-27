FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Headless pygame (no display / sound card in the container)
    SDL_VIDEODRIVER=dummy \
    SDL_AUDIODRIVER=dummy \
    # Cloud Run sends traffic to port 8080
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=8080 \
    MODELS_DIR=/app/models

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# App code and trained model(s)
COPY robo_dunk/ robo_dunk/
COPY sandbox.py .
COPY models/ models/

EXPOSE 8080

CMD ["python", "sandbox.py"]
