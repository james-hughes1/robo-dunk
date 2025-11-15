FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and pygame
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libsdl2-2.0-0 \
    libsdl2-image-2.0-0 \
    libsdl2-mixer-2.0-0 \
    libsdl2-ttf-2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Turn off streamlit email message
RUN mkdir -p /root/.streamlit
COPY .streamlit/credentials.toml /root/.streamlit/credentials.toml

# Set environment variables for headless operation
ENV SDL_VIDEODRIVER=dummy
ENV SDL_AUDIODRIVER=dummy
ENV DISPLAY=:99

# Create models directory
RUN mkdir -p /app/models

# Expose ports
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the app
CMD ["streamlit", "run", "sandbox.py", "--server.port=8501", "--server.address=0.0.0.0"]
