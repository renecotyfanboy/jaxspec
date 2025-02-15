# syntax=docker/dockerfile:1
FROM renecoty/heasoft:v6.34

WORKDIR /jaxspec

USER root
# Project initialization:
RUN apt-get update && \
    apt-get install -y git

RUN pip install uv

ADD src /jaxspec/src
ADD tests /jaxspec/tests
ADD pyproject.toml /jaxspec/pyproject.toml
ADD README.md /jaxspec/README.md

RUN uv sync --no-group docs --no-group dev --python 3.12
ENV PATH="/app/.venv/bin:$PATH"