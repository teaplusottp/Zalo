# Use PyTorch as base image
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

# Set working directory
WORKDIR /code

# Update apt and install system dependencies
RUN apt-get update && apt-get -y install \
    libgl1-mesa-glx \
    libglib2.0-0 \
    vim \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy source code from current directory
COPY . /code/

# Install Python dependencies
RUN pip install --no-cache-dir jupyterlab && \
    pip install --no-cache-dir -r /code/requirements.txt && \
    pip install --no-cache-dir numpy

# Create result directory
RUN mkdir -p /result

# Expose ports
# 9777 for JupyterLab
EXPOSE 9777

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Default command: Start bash
CMD ["/bin/bash"]
