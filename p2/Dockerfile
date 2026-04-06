# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    PYTHONPATH=/workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY p2/requirements.txt /tmp/requirements.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    -r /tmp/requirements.txt

COPY p2 /workspace/p2

EXPOSE 8000

CMD ["uvicorn", "p2.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
