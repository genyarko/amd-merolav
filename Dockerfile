# Multi-stage Dockerfile for rocm-migrate
# Supports both CPU-only (rule-based migration) and GPU (full validation on ROCm)

# ==============================================================================
# Stage 1: CPU-only image (lightweight, no ROCm runtime needed)
# ==============================================================================
FROM python:3.12-slim AS cpu

LABEL maintainer="George Nyarko"
LABEL description="CUDA-to-ROCm migration tool — rule-based + LLM-powered"
LABEL org.opencontainers.image.source="https://github.com/genyarko/amd-merolav"

WORKDIR /app

# Install system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python package
COPY pyproject.toml README.md ./
COPY cli/ cli/
COPY core/ core/
COPY agents/ agents/
COPY config/ config/
COPY knowledge/ knowledge/
COPY testing/ testing/
COPY __main__.py ./

RUN pip install --no-cache-dir -e ".[all]"

# Default entrypoint
ENTRYPOINT ["rocm-migrate"]
CMD ["--help"]

# ==============================================================================
# Stage 2: GPU image with ROCm runtime (for real validation on AMD GPUs)
# ==============================================================================
FROM rocm/pytorch:latest AS gpu

LABEL maintainer="George Nyarko"
LABEL description="CUDA-to-ROCm migration tool with ROCm GPU validation support"
LABEL org.opencontainers.image.source="https://github.com/genyarko/amd-merolav"

WORKDIR /app

# Copy and install Python package
COPY pyproject.toml README.md ./
COPY cli/ cli/
COPY core/ core/
COPY agents/ agents/
COPY config/ config/
COPY knowledge/ knowledge/
COPY testing/ testing/
COPY __main__.py ./

RUN pip install --no-cache-dir -e ".[all]"

# Default entrypoint
ENTRYPOINT ["rocm-migrate"]
CMD ["--help"]
