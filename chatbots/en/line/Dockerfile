
FROM python:3.14.4-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_FILE=main.py

WORKDIR /app

# Install uv to resolve dependencies from uv.lock reproducibly.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

COPY pyproject.toml uv.lock main.py ./
RUN uv sync --frozen --no-dev --no-install-project

COPY data/ ./data

CMD ["sh", "-c", "./.venv/bin/python ${APP_FILE}"]