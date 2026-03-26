# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Dockerfile for Log Anomaly Investigation Environment
# Build from project root:
# docker build -t log-anomaly-env:latest -f server/Dockerfile .

# Accept base image as build argument for CI/CD flexibility
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

# Set working directory
WORKDIR /app

# Install system dependencies for log analysis
RUN apt-get update && apt-get install -y --no-install-recommends \
    grep \
    gawk \
    sed \
    coreutils \
    procps \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Copy environment code
COPY . /app

# Install the package in development mode
RUN pip install --no-cache-dir -e /app

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:$PYTHONPATH
ENV LOG_LEVEL=INFO

# Create directories for outputs
RUN mkdir -p /app/outputs/logs /app/outputs/evals

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
