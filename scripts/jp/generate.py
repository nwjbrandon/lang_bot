#!/usr/bin/env python3
"""
ollama pull qwen3:8b
ollama serve
python generate.py
"""

import json

import pandas as pd
import requests

INPUT_CSV = "data/raw/jlpt_n2_vocab_mori_no_nihongo.csv"
OUTPUT_CSV = "data/csv/jlpt_n2_vocab_mori_no_nihongo.csv"
MODEL = "qwen3:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"

SCHEMA = {"type": "object", "properties": {"meaning": {"type": "string"}, "sentence": {"type": "string"}}, "required": ["meaning", "sentence"], "additionalProperties": False}

df = pd.read_csv(INPUT_CSV)
session = requests.Session()


def clean(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def generate_entry(kanji: str, hiragana: str) -> tuple[str, str]:
    prompt = f"""
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

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": SCHEMA,
        "think": False,
        "options": {"temperature": 0.2},
        "keep_alive": "10m",
    }

    r = session.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    result = r.json()

    text = result.get("response", "")
    data = json.loads(text)
    meaning = str(data["meaning"]).strip()
    sentence = str(data["sentence"]).strip()
    return meaning, sentence


meanings = []
sentences = []

for i, row in df.iterrows():
    kanji = clean(row.get("Kangi"))
    hiragana = clean(row.get("Hiragana"))

    try:
        meaning, sentence = generate_entry(kanji, hiragana)
    except Exception as e:
        print(f"ERROR on row {i + 1}: {e}")
        meaning, sentence = "", ""

    meanings.append(meaning)
    sentences.append(sentence)
    print(f"Processed {i + 1}/{len(df)}: {kanji} ({hiragana}) -> {meaning} / {sentence}")

df["Meaning"] = meanings
df["Sentence"] = sentences
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved to {OUTPUT_CSV}")
