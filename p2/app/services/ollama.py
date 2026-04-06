from __future__ import annotations

from typing import Any

import httpx

from p2.app.config import DEFAULT_LLM_MODEL, OLLAMA_BASE_URL, OLLAMA_TIMEOUT_SECONDS


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama service cannot be reached."""


class OllamaClient:
    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        timeout_seconds: float = OLLAMA_TIMEOUT_SECONDS,
        default_model: str = DEFAULT_LLM_MODEL,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.default_model = default_model

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.request(method, url, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as exc:
            raise OllamaUnavailableError(str(exc)) from exc

    def health(self) -> dict[str, Any]:
        payload = self._request("GET", "/api/tags")
        models = [item.get("name", "") for item in payload.get("models", []) if item.get("name")]
        return {
            "status": "ok",
            "ollama_available": True,
            "default_model": self.default_model,
            "installed_models": models,
        }

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        options = {"temperature": temperature}
        payload: dict[str, Any] = {
            "model": model or self.default_model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system:
            payload["system"] = system
        data = self._request("POST", "/api/generate", payload)
        return {
            "model": data.get("model", payload["model"]),
            "response": data.get("response", ""),
            "done": bool(data.get("done", False)),
            "total_duration": data.get("total_duration"),
            "eval_count": data.get("eval_count"),
        }
