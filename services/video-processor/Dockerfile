FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app

# System dependencies, Python 3.11, pip, and build tools
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-venv \
    build-essential \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set python and python3 to python3.11, and upgrade pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    python3 -m pip install --upgrade pip

# Copy requirements and install dependencies with pip
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Create needed directories and download YOLOv8n model
RUN mkdir -p /app/models && \
    curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -o /app/models/yolov8n.pt && \
    mkdir -p /data/videos /data/processed /app/logs

# Create a non-root user and set ownership
RUN addgroup --system app && adduser --system --group app
RUN chown -R app:app /app

# Switch to the non-root user
USER app

# CUDA and NVIDIA environment variables for GPU access
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Run your worker
CMD ["python3", "worker.py"]
