# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm
WORKDIR /workspace

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/root/.cache/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY run.py .
ENTRYPOINT ["python", "run.py"]
