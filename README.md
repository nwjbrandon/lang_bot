# Lang Bot

Quiz bots for language study. A single shared framework (`langbot/`) powers
several bots that differ only in **subject** (what's being quizzed) and
**transport** (LINE or Telegram):

| App module                | Subject              | Transport         |
| ------------------------- | -------------------- | ----------------- |
| `langbot.apps.jp_telegram`| JLPT vocab           | Telegram (polling)|
| `langbot.apps.jp_line`    | JLPT vocab           | LINE (webhook)    |
| `langbot.apps.en_line`    | English/Japanese     | LINE (webhook)    |

## Layout

```
langbot/
  loader.py          CSV -> entries
  quiz.py            mode resolution + 4-option question generation
  session.py         per-user spaced-repetition state + controller
  engine.py          transport-agnostic command/action dispatch
  present.py         build renderer-agnostic views
  subjects/          jlpt_vocab, en_ja_phrases (content + localized strings)
  transports/        line, telegram, render (plain/HTML)
  apps/              thin entrypoints (one per bot)
chatbots/<lang>/<transport>/   per-bot deploy files (docker-compose.yml, .env)
scripts/jp/generate.py         Ollama helper to fill meaning/sentence columns
```

## Installation

Install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or
pip install uv
```

## Getting started

Install dependencies:

```bash
uv sync            # production
uv sync --dev      # with dev tools (ruff)
```

Configure environment (see each `chatbots/**/.local.env` for the variables a
bot needs) and run an app:

```bash
# JLPT vocab on LINE
CSV_PATH=./data/jp/csv LINE_CHANNEL_ACCESS_TOKEN=... LINE_CHANNEL_SECRET=... \
  uv run python -m langbot.apps.jp_line

# JLPT vocab on Telegram
CSV_PATH=./data/jp/csv TELEGRAM_BOT_TOKEN=... \
  uv run python -m langbot.apps.jp_telegram

# English/Japanese phrases on LINE
CSV_PATH=./data/en/csv LINE_CHANNEL_ACCESS_TOKEN=... LINE_CHANNEL_SECRET=... \
  uv run python -m langbot.apps.en_line
```

### Environment variables

| Variable                     | Used by   | Notes                                              |
| ---------------------------- | --------- | -------------------------------------------------- |
| `CSV_PATH`                   | all       | CSV file or directory of CSVs (default per app)    |
| `QUIZ_MODE`                  | all       | optional starting mode (e.g. `MODE_AUTO`)          |
| `TELEGRAM_BOT_TOKEN`         | telegram  | required                                           |
| `LINE_CHANNEL_ACCESS_TOKEN`  | line      | required                                           |
| `LINE_CHANNEL_SECRET`        | line      | required (webhook signature verification)          |
| `LINE_HOST` / `LINE_PORT`    | line      | default `0.0.0.0:8000`                              |

CSV files need the columns for their subject (JLPT: `Kangi`, `Hiragana`, and
optionally `Meaning`, `Sentence`; EN/JA: `English`, `Japanese`). Column matching
is case-insensitive.

## Docker

Each bot deploys independently from its directory. The compose files build the
shared image from the repo root and select the app via `APP_MODULE`:

```bash
cd chatbots/jp/line          # or jp/telegram, en/line
cp .local.env .env           # then fill in real secrets
# place CSVs under ./data (mounted read-only at /app/data)
docker compose up -d --build
```

## Linting

```bash
./lint.sh        # ruff check + import sort + format
```

## Data generation

`scripts/jp/generate.py` fills `Meaning`/`Sentence` columns with a local Ollama
model:

```bash
ollama pull qwen3:8b && ollama serve
uv run python scripts/jp/generate.py --input data/raw/in.csv --output data/csv/out.csv
```
