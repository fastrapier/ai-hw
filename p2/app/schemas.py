from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class PredictionItem(BaseModel):
    index: int
    label: str
    score: float
    text_preview: str


class PredictRequest(BaseModel):
    text: Optional[str] = Field(default=None, description="Single text to classify.")
    texts: Optional[List[str]] = Field(default=None, description="Batch of texts.")

    @model_validator(mode="after")
    def validate_payload(self) -> "PredictRequest":
        has_text = self.text is not None
        has_texts = self.texts is not None

        if has_text == has_texts:
            raise ValueError("Provide exactly one of 'text' or 'texts'.")

        if self.text is not None and not self.text.strip():
            raise ValueError("'text' must not be empty.")

        if self.texts is not None:
            cleaned = [item.strip() for item in self.texts]
            if not cleaned or any(not item for item in cleaned):
                raise ValueError("'texts' must contain non-empty strings.")
            self.texts = cleaned

        if self.text is not None:
            self.text = self.text.strip()

        return self


class PredictResponse(BaseModel):
    model: str
    count: int
    results: List[PredictionItem]


class HealthResponse(BaseModel):
    status: str
    sentiment_model: str
    sentiment_model_loaded: bool


class LlmHealthResponse(BaseModel):
    status: str
    ollama_available: bool
    default_model: str
    installed_models: List[str]


class LlmGenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    model: Optional[str] = None
    system: Optional[str] = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class LlmGenerateResponse(BaseModel):
    model: str
    response: str
    done: bool
    total_duration: Optional[int] = None
    eval_count: Optional[int] = None


class AgentRequest(BaseModel):
    question: str = Field(min_length=1)
    model: Optional[str] = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class AgentResponse(BaseModel):
    model: str
    answer: str
    selected_tools: List[str]
    context_sources: List[str]
