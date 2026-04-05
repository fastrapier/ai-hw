"""Sentiment analysis: Hugging Face pipeline (multilingual XLM-R)."""
from __future__ import annotations

import json
import os
from pathlib import Path

from transformers import pipeline

INPUT_DIR = Path(os.environ.get("INPUT_DIR", "/workspace/input"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/workspace/output"))
MODEL_ID = os.environ.get("SENTIMENT_MODEL", "cardiffnlp/twitter-xlm-roberta-base-sentiment")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    texts: list[str] = []
    for p in sorted(INPUT_DIR.glob("*.txt")):
        raw = p.read_text(encoding="utf-8").strip()
        if raw:
            texts.append(raw)
    if not texts:
        texts = ["The product is great and I love it."]

    clf = pipeline(
        "sentiment-analysis",
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        truncation=True,
        max_length=512,
    )
    results = []
    for i, text in enumerate(texts):
        out = clf(text)[0]
        results.append(
            {
                "index": i,
                "text_preview": text[:200] + ("…" if len(text) > 200 else ""),
                "label": out["label"],
                "score": float(out["score"]),
            }
        )

    payload = {"model": MODEL_ID, "results": results}
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
