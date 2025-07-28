FROM --platform=linux/amd64 python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements_docker.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_docker.txt

# Copy the rest of the application
COPY . .

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Set environment variables for input/output
ENV INPUT_DIR=/app/input \
    OUTPUT_DIR=/app/output

# Set the entrypoint
ENTRYPOINT ["python", "main.py"]
