# arm64 image for M-series Macs running Docker Desktop.
#
# Apple Silicon Docker can run arm64 images natively. MPS itself is not
# exposed inside Docker (the GPU passthrough story for MPS via Docker is
# not there yet), so this image runs the eval on CPU. It's here for parity
# with the Linux/CUDA image so a reviewer can confirm the package layout
# is right on their Mac without installing Python locally.
#
# For real M3 / MPS measurements, install with uv directly on the host
# (see docs/M3_NOTES.md). This image is for parity and for the
# benchmark.yml replay step under arm64.

FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.hf_home

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential \
        gdal-bin libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src

# Skip CUDA-flavored extras here; install only the dev tools and let a
# reviewer add `[ttm]` or `[prithvi]` themselves if they need to load weights.
RUN uv venv --python 3.12 && uv pip install -e ".[dev,live]"

COPY . /app

ENTRYPOINT ["uv", "run", "riprap-models"]
CMD ["device"]
