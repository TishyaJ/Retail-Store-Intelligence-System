FROM python:3.11-slim

WORKDIR /app

# System dependencies for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (specific index URL)
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    torchvision==0.18.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements
COPY pipeline/requirements.txt .
RUN grep -v "torch==" requirements.txt | grep -v "torchvision==" \
    | pip install --no-cache-dir -r /dev/stdin

# Copy all source files
COPY pipeline/ ./pipeline/
COPY store_layout/ ./store_layout/
COPY POS_transactions.csv ./POS_transactions.csv

# Make run.sh executable
RUN chmod +x /app/pipeline/run.sh

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import onnxruntime; print('ok')" || exit 1

CMD ["bash", "/app/pipeline/run.sh"]
