FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_MODULE=langbot.apps.jp_line

WORKDIR /app

# Install uv to resolve dependencies from uv.lock reproducibly.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY langbot ./langbot

# APP_MODULE selects which bot to run (overridden per docker-compose service).
# CSV data is provided at runtime via a mounted volume (see compose files).
CMD ["sh", "-c", ".venv/bin/python -m ${APP_MODULE}"]
