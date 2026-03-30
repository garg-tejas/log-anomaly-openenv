# Dockerfile for Log Anomaly Investigation Environment
# OpenEnv AI Hackathon Submission
#
# Build: docker build -t log-anomaly-env:latest .
# Run:   docker run -p 8000:8000 log-anomaly-env:latest

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for log analysis commands
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

# Copy all project files
COPY . /app

# Install the package
RUN pip install --no-cache-dir -e /app

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:$PYTHONPATH
ENV LOG_LEVEL=INFO

# Create directories for outputs
RUN mkdir -p /app/outputs/logs /app/outputs/evals

# Expose port (HF Spaces expects 7860 by default, but we use 8000)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
