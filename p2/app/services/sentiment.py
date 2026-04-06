from __future__ import annotations

from collections.abc import Callable
from typing import Any

from p2.app.config import DEFAULT_SENTIMENT_MODEL


PipelineFactory = Callable[[], Any]


def default_pipeline_factory(model_id: str) -> Callable[[str | list[str]], list[dict[str, Any]] | dict[str, Any]]:
    from transformers import pipeline

    return pipeline(
        "sentiment-analysis",
        model=model_id,
        tokenizer=model_id,
        truncation=True,
        max_length=512,
    )


class SentimentService:
    def __init__(
        self,
        model_id: str = DEFAULT_SENTIMENT_MODEL,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        self.model_id = model_id
        self._pipeline_factory = pipeline_factory or (lambda: default_pipeline_factory(model_id))
        self._pipeline: Callable[[str | list[str]], list[dict[str, Any]] | dict[str, Any]] | None = None

    @property
    def model_loaded(self) -> bool:
        return self._pipeline is not None

    def _get_pipeline(self) -> Callable[[str | list[str]], list[dict[str, Any]] | dict[str, Any]]:
        if self._pipeline is None:
            self._pipeline = self._pipeline_factory()
        return self._pipeline

    def analyze(self, texts: list[str]) -> list[dict[str, Any]]:
        raw_results = self._get_pipeline()(texts)
        if isinstance(raw_results, dict):
            raw_results = [raw_results]

        results: list[dict[str, Any]] = []
        for index, (text, item) in enumerate(zip(texts, raw_results, strict=True)):
            results.append(
                {
                    "index": index,
                    "label": str(item["label"]),
                    "score": float(item["score"]),
                    "text_preview": text[:200] + ("..." if len(text) > 200 else ""),
                }
            )
        return results
