# Use Python 3.11 slim image for a compact container footprint
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies needed for installing python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    liblmdb-dev \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies in a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Final runner stage
FROM python:3.11-slim AS runner

WORKDIR /app

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Install runtime LMDB shared libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    liblmdb0 \
    && rm -rf /var/lib/apt/lists/*

# Copy DocForge project files
COPY config.yaml .
COPY src/ ./src/

EXPOSE 8000

# Default execution entrypoint starts the API backend and Web interface
CMD ["python", "src/deployment/api/main.py"]
