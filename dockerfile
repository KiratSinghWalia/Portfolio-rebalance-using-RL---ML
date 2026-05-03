FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy

WORKDIR /workspace

# System dependencies:
# - libgomp1: OpenMP runtime needed by LightGBM on Linux
# - build-essential/cmake: useful if any package needs native build
# - git/curl/ca-certificates: common dev tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files first for better Docker caching
COPY pyproject.toml ./
COPY uv.lock* ./

# Install dependencies into project .venv
RUN uv sync --python 3.12 || uv lock && uv sync --python 3.12

# Copy the repo
COPY . .

CMD ["/bin/bash"]