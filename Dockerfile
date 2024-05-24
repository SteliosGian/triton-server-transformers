FROM nvcr.io/nvidia/tritonserver:24.04-pyt-python-py3

ARG APPDIR=/ner-model

WORKDIR $APPDIR

ARG POETRY_VERSION=1.8.3

ENV POETRY_NO_INTERACTION=1 \
    # Don't create virtual env for poetry
    POETRY_VIRTUALENVS_CREATE=0 \
    # No cache for pip
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Send stdout and stderr streams to terminal
    PYTHONUNBUFFERED=1 \
    # Increase timeout for long-running packages like torch
    PIP_DEFAULT_TIMEOUT=3000 \
    HF_HOME=${APPDIR}/cache/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir poetry==${POETRY_VERSION}

COPY pyproject.toml poetry.lock ./

RUN poetry install --only main --no-root --no-cache

COPY models models

RUN addgroup --group appgroup && adduser --uid 2000 --ingroup appgroup appuser

RUN chown -R appuser:appgroup $APPDIR

USER 2000
