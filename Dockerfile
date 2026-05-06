# Multi-stage build for CogniVault RAG System
FROM python:3.11-slim as builder

WORKDIR /build

# Install dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (if using pip)
COPY pyproject.toml* ./
RUN pip install --no-cache-dir uv

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies from builder
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv
COPY pyproject.toml* ./

# Install Python dependencies
RUN if [ -f pyproject.toml ]; then \
    uv pip install --system -r <(uv pip compile pyproject.toml) && \
    uv pip install --system -e . ; \
    else \
    pip install --no-cache-dir fastapi uvicorn langchain-openai langchain-qdrant qdrant-client langchain-text-splitters python-dotenv ; \
    fi

# Copy application code
COPY main.py app.py ./
COPY pdf_loader.py retreive.py ./
COPY market_frontend.html ./

# Create necessary directories for persistence
RUN mkdir -p /app/pdfs /app/qdrant_local_data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "app.py"]
