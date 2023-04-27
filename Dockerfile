# syntax=docker/dockerfile:experimental
FROM python:3.9-buster

# upgrade pip
RUN python -m pip install --upgrade pip

# install poetry
RUN pip install --upgrade pip \
    && pip install poetry==1.1.15

# install dependencies
# WORKDIR /tmp
# COPY ./poetry.lock /tmp/
# COPY ./pyproject.toml /tmp/

RUN poetry config virtualenvs.create true
# RUN poetry config virtualenvs.create true \
#     && poetry install \
#     && rm poetry.lock \
#     && rm pyproject.toml
