# Use a lightweight Python base image with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    libgl1 \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy all files from the project to the container
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && pip install \
    torch torchvision \
    diffusers transformers peft opencv-python protobuf sentencepiece

# Expose port for debugging (optional, if needed for local testing)
EXPOSE 8080

# Command to run the Replicate server
CMD ["replicate", "serve"]
