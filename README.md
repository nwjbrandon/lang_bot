# Lang Bot

This repository contains multiple quiz bots for language study:

## 1. Installation

### Install uv

```bash
curl -sSf https://install.uv.dev | sh
# Or with pip
pip install uv
```


## 2. Getting Started

### Install dependencies

```bash
# For development
uv sync --dev
# For production
uv sync
```

### Run application

```bash
python main.py
```

- [chatbots/jp/telegram/main.py](chatbots/jp/telegram/main.py): JLPT vocab quiz (Telegram)
- [chatbots/jp/line/main.py](chatbots/jp/line/main.py): JLPT vocab quiz (LINE)
- [chatbots/en/line/main.py](chatbots/en/line/main.py): English phrase quiz (LINE)
