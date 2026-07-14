# syntax=docker/dockerfile:1
#
# LocalVectorDB server image.
#
# Builds from source in this repo (not from PyPI) so the image is validated in CI
# against the code being changed, and so it works before a release is published.
#
#   docker build -t localvectordb:local .
#   docker run --rm -p 8000:8000 -v lvdb-data:/data localvectordb:local
#
# Databases live in the /data volume (LVDB_DATABASE_ROOT_DIR). Configure via
# LVDB_* environment variables, or mount a TOML config and point at it:
#
#   docker run --rm -p 8000:8000 -v $PWD/config.toml:/etc/lvdb/config.toml:ro \
#     -v lvdb-data:/data localvectordb:local \
#     lvdb --config /etc/lvdb/config.toml serve --host 0.0.0.0 --port 8000

# ----------------------------------------------------------------------------
# Builder: resolve dependencies into a self-contained virtualenv.
# ----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /src

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# hatchling reads project.readme, so README.md must be present to build the wheel.
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# The `server` extra pulls in FastAPI/uvicorn. File extraction is deliberately not
# installed: it is heavy and most deployments do not upload documents through the
# API. Add `[server,file-extraction]` here if you need upload/extraction.
RUN pip install ".[server]"

# ----------------------------------------------------------------------------
# Runtime: copy only the venv. No build toolchain, no source tree, non-root.
# ----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LVDB_DATABASE_ROOT_DIR=/data

# `lvdb serve` pre-flights a local Ollama install and exits non-zero when it is
# missing. That is a developer convenience and is wrong in a container: the embedding
# provider is configured through LVDB_EMBEDDING_* and may be OpenAI, or an Ollama on
# another host. Without this the image would refuse to start at all.
ENV LVDB_DISABLE_OLLAMA_CHECK=true

# Run as an unprivileged user; own /data so the volume is writable without root.
RUN useradd --create-home --uid 10001 lvdb \
    && mkdir -p /data \
    && chown -R lvdb:lvdb /data

COPY --from=builder --chown=lvdb:lvdb /opt/venv /opt/venv

USER lvdb
WORKDIR /home/lvdb

VOLUME ["/data"]
EXPOSE 8000

# Routes are mounted under the versioned API prefix, so the health path is
# /api/v1/health -- NOT /health, which 404s. The endpoint probes Ollama with a bounded
# 2s timeout and still reports "healthy" when Ollama is absent, but the first request
# also pays import/init warm-up, hence the generous timeout and start-period.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=8).status == 200 else 1)"

CMD ["lvdb", "serve", "--host", "0.0.0.0", "--port", "8000"]
