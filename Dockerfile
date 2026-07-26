ARG CUDA_IMAGE=nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/cache/huggingface \
    MAX_JOBS=2 \
    PATH=/opt/forge-venv/bin:${PATH} \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        g++ \
        ninja-build \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-gpu.txt pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python3 -m venv /opt/forge-venv \
    && python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-gpu.txt \
    && python -m pip install --no-cache-dir --no-deps .

RUN forge-engine --version

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

ENTRYPOINT ["forge-engine"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
