# Cumulus Job Container
# Base image with Python and OpenCL support
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # OpenCL support
    ocl-icd-opencl-dev \
    clinfo \
    # Build tools for Python packages
    build-essential \
    # Utilities
    unzip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install common ML libraries (optional - can be in requirements.txt)
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    jupyter \
    requests

# Create app directory
WORKDIR /app

# Copy job execution scripts
COPY entrypoint.sh /app/entrypoint.sh
COPY runner.py /app/runner.py

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Create job directory
RUN mkdir -p /job

# Set working directory to job
WORKDIR /job

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/job

# Default entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]