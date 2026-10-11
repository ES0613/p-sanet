FROM python:3.11.15-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libxrender1 libxext6 libsm6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY requirements-lock.txt requirements-fast-transformers.txt pyproject.toml README.md LICENSE MANIFEST.in ./
COPY src ./src
COPY tools ./tools
RUN python -m pip install --upgrade pip==25.0.1 \
    && python -m pip install -r requirements-lock.txt \
    && python tools/install_fast_transformers.py \
    && python -m pip install --no-deps .

ENTRYPOINT ["physaft-esper-pvap"]
CMD ["--help"]
