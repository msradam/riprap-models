# Linux x86_64 / CUDA evaluation image.
#
# This image installs every model extra and the dev extras so a reviewer
# can run the full eval + benchmark pipeline against an NVIDIA GPU. It
# pins the same transformers + huggingface_hub versions used in
# riprap-nyc/Dockerfile so adapter loading does not silently regress.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.hf_home

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential \
        python3.12 python3.12-dev python3.12-venv \
        gdal-bin libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src

RUN uv venv --python 3.12 && \
    uv pip install -e ".[dev,terramind,prithvi,ttm,live]"

COPY . /app

# Default: print the device + dtype the runtime picks. Override with
# `docker run ... riprap-models eval terramind-buildings`.
ENTRYPOINT ["uv", "run", "riprap-models"]
CMD ["device"]
