FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir uv

COPY . .

RUN uv sync --frozen --no-dev

ENV PYTHONPATH=/app/src:$PYTHONPATH

# Expose API port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1


CMD ["uv", "run", "python", "run_api_server.py"]

