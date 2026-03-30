# Dockerfile for Log Anomaly Investigation Environment
# OpenEnv AI Hackathon Submission
#
# Build: docker build -t log-anomaly-env:latest .
# Run:   docker run -p 8000:8000 log-anomaly-env:latest

FROM python:3.11-slim

# Install system dependencies for log analysis commands
RUN apt-get update && apt-get install -y --no-install-recommends \
    grep \
    gawk \
    sed \
    coreutils \
    procps \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create user for HuggingFace Spaces (runs as user ID 1000)
RUN useradd -m -u 1000 user
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO

WORKDIR $HOME/app

# Copy requirements first for better caching
COPY --chown=user requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Copy all project files
COPY --chown=user . $HOME/app

# Install the package
RUN pip install --no-cache-dir -e $HOME/app

# Create directories for outputs
RUN mkdir -p $HOME/app/outputs/logs $HOME/app/outputs/evals

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
