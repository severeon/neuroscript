# NeuroScript v2 Runtime Docker Image
# This image provides a complete PyTorch environment for executing
# generated NeuroScript models with GPU support.

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install additional Python dependencies
RUN pip install --no-cache-dir \
    pyyaml \
    yamale \
    psutil \
    numpy

# Set environment variables for optimal PyTorch performance
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/tmp/torch

# Create directory for PyTorch cache
RUN mkdir -p /tmp/torch

# Default command (will be overridden by container runtime)
CMD ["python", "--version"]
