# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# System libraries required by OpenCV / Pillow, plus curl for the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Keep Python quiet and unbuffered, and route the HF cache to a stable path
# that docker-compose mounts as a named volume so models persist across builds.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    STIX_DEVICE=cpu

# Install the CPU-only PyTorch wheels first. These are far smaller than the
# default CUDA builds and are what this image will actually run. requirements
# then resolves torch/torchvision as already-satisfied.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Temp working dirs (the app also creates these at runtime under the system temp).
RUN mkdir -p /tmp/stix_app/uploads /tmp/stix_app/output /tmp/stix_app/masks

EXPOSE 8000

# Report unhealthy until the model has finished loading and /health responds.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
