#!/usr/bin/env bash
set -euo pipefail
uv run ruff check
uv run ruff check --select I --fix
uv run ruff format
