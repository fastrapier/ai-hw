from __future__ import annotations

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from p2.app.dependencies import get_agent_service, get_ollama_client, get_sentiment_service
from p2.app.main import app


class FakeSentimentService:
    model_id = "fake-sentiment-model"
    model_loaded = True

    def analyze(self, texts: list[str]) -> list[dict[str, object]]:
        return [
            {
                "index": index,
                "label": "positive" if "good" in text.lower() or "полез" in text.lower() else "negative",
                "score": 0.99,
                "text_preview": text[:200],
            }
            for index, text in enumerate(texts)
        ]


class FakeOllamaClient:
    default_model = "qwen2.5:3b"

    def health(self) -> dict[str, object]:
        return {
            "status": "ok",
            "ollama_available": True,
            "default_model": self.default_model,
            "installed_models": ["qwen2.5:3b", "llama3.2:3b"],
        }

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.2,
    ) -> dict[str, object]:
        return {
            "model": model or self.default_model,
            "response": f"stubbed response for: {prompt[:30]}",
            "done": True,
            "total_duration": 123,
            "eval_count": 42,
        }


class FakeAgentService:
    def answer(self, question: str, model: str | None = None, temperature: float = 0.2) -> dict[str, object]:
        return {
            "model": model or "qwen2.5:3b",
            "answer": f"agent answer for: {question}",
            "selected_tools": ["project_report", "llm_evaluation"],
            "context_sources": ["p2/report.md", "p2/eval/results.md"],
        }


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    app.dependency_overrides[get_sentiment_service] = lambda: FakeSentimentService()
    app.dependency_overrides[get_ollama_client] = lambda: FakeOllamaClient()
    app.dependency_overrides[get_agent_service] = lambda: FakeAgentService()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["sentiment_model"] == "fake-sentiment-model"


def test_predict_single_text(client: TestClient) -> None:
    response = client.post("/predict", json={"text": "This is a good API"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "fake-sentiment-model"
    assert payload["count"] == 1
    assert payload["results"][0]["label"] == "positive"


def test_predict_batch(client: TestClient) -> None:
    response = client.post("/predict", json={"texts": ["good result", "bad result"]})
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    assert payload["results"][1]["label"] == "negative"


def test_predict_rejects_empty_payload(client: TestClient) -> None:
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_llm_health(client: TestClient) -> None:
    response = client.get("/llm/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ollama_available"] is True
    assert "qwen2.5:3b" in payload["installed_models"]


def test_llm_generate(client: TestClient) -> None:
    response = client.post("/llm/generate", json={"prompt": "hello", "model": "qwen2.5:3b"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "qwen2.5:3b"
    assert payload["done"] is True


def test_agent_answer(client: TestClient) -> None:
    response = client.post("/agent/answer", json={"question": "Какую модель выбрали?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "qwen2.5:3b"
    assert payload["selected_tools"] == ["project_report", "llm_evaluation"]
