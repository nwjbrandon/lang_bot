#!/usr/bin/env python3
"""Generate JLPT meaning/sentence columns for a vocab CSV using a local Ollama model.

Setup:
    ollama pull qwen3:8b
    ollama serve
    python scripts/jp/generate.py \
        --input data/raw/jlpt_n2_vocab_mori_no_nihongo.csv \
        --output data/csv/jlpt_n2_vocab_mori_no_nihongo.csv
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import pandas as pd
import requests

DEFAULT_INPUT = "data/raw/jlpt_n2_vocab_mori_no_nihongo.csv"
DEFAULT_OUTPUT = "data/csv/jlpt_n2_vocab_mori_no_nihongo.csv"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"

SCHEMA = {
    "type": "object",
    "properties": {"meaning": {"type": "string"}, "sentence": {"type": "string"}},
    "required": ["meaning", "sentence"],
    "additionalProperties": False,
}

PROMPT_TEMPLATE = """
You are creating JLPT N2 vocabulary study data.

Target word:
Kanji: {kanji}
Hiragana: {hiragana}

Return ONLY valid JSON in this exact format:
{{
    "meaning": "short natural English meaning",
    "sentence": "natural Japanese sentence using the target word"
}}

Rules:
- Use simple, clear English for a JLPT learner.
- Keep meaning concise (about 2-7 words).
- Sentence length: about 12-28 Japanese characters.
- The sentence must sound natural in daily Japanese.
- Use the target word exactly once in the sentence.

Return only the JSON object.
"""


def clean(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def generate_entry(session: requests.Session, url: str, model: str, kanji: str, hiragana: str) -> Tuple[str, str]:
    payload = {
        "model": model,
        "prompt": PROMPT_TEMPLATE.format(kanji=kanji, hiragana=hiragana),
        "stream": False,
        "format": SCHEMA,
        "think": False,
        "options": {"temperature": 0.2},
        "keep_alive": "10m",
    }

    response = session.post(url, json=payload, timeout=120)
    response.raise_for_status()

    data = json.loads(response.json().get("response", ""))
    return str(data["meaning"]).strip(), str(data["sentence"]).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV with Kangi/Hiragana columns.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name.")
    parser.add_argument("--url", default=DEFAULT_OLLAMA_URL, help="Ollama generate endpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    session = requests.Session()

    meanings, sentences = [], []
    for i, row in df.iterrows():
        kanji = clean(row.get("Kangi"))
        hiragana = clean(row.get("Hiragana"))

        try:
            meaning, sentence = generate_entry(session, args.url, args.model, kanji, hiragana)
        except Exception as exc:  # noqa: BLE001 - best-effort per row; keep going
            print(f"ERROR on row {i + 1}: {exc}")
            meaning, sentence = "", ""

        meanings.append(meaning)
        sentences.append(sentence)
        print(f"Processed {i + 1}/{len(df)}: {kanji} ({hiragana}) -> {meaning} / {sentence}")

    df["Meaning"] = meanings
    df["Sentence"] = sentences

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
