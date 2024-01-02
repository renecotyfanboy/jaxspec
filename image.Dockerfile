# syntax=docker/dockerfile:1
FROM renecoty/heasoft:poetry

WORKDIR /jaxspec

USER root
# Project initialization:
RUN apt-get update && \
    apt-get install -y git

RUN pip install poetry

COPY pyproject.toml poetry.lock /jaxspec/

ENV POETRY_VIRTUALENVS_IN_PROJECT=0 \
    POETRY_VIRTUALENVS_CREATE=0 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

RUN poetry install --no-root --no-ansi --no-interaction

COPY . /jaxspec

RUN poetry install --no-ansi --no-interaction

ENTRYPOINT ["poetry", "run", "pytest"]