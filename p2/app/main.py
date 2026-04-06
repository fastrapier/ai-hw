from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException

from p2.app.config import DEFAULT_LLM_MODEL
from p2.app.dependencies import get_agent_service, get_ollama_client, get_sentiment_service
from p2.app.schemas import (
    AgentRequest,
    AgentResponse,
    HealthResponse,
    LlmGenerateRequest,
    LlmGenerateResponse,
    LlmHealthResponse,
    PredictRequest,
    PredictResponse,
)
from p2.app.services.agent import StudyAgentService
from p2.app.services.ollama import OllamaClient, OllamaUnavailableError
from p2.app.services.sentiment import SentimentService


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.sentiment_service = SentimentService()
    app.state.ollama_client = OllamaClient()
    app.state.agent_service = StudyAgentService(app.state.ollama_client)
    yield


app = FastAPI(
    title="p2 ml and llm api",
    description="FastAPI service for sentiment analysis, Ollama access and a simple study agent.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health(sentiment_service: SentimentService = Depends(get_sentiment_service)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        sentiment_model=sentiment_service.model_id,
        sentiment_model_loaded=sentiment_service.model_loaded,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(
    payload: PredictRequest,
    sentiment_service: SentimentService = Depends(get_sentiment_service),
) -> PredictResponse:
    texts = [payload.text] if payload.text is not None else payload.texts or []
    results = sentiment_service.analyze(texts)
    return PredictResponse(model=sentiment_service.model_id, count=len(results), results=results)


@app.get("/llm/health", response_model=LlmHealthResponse)
def llm_health(ollama_client: OllamaClient = Depends(get_ollama_client)) -> LlmHealthResponse:
    try:
        data = ollama_client.health()
    except OllamaUnavailableError:
        return LlmHealthResponse(
            status="degraded",
            ollama_available=False,
            default_model=DEFAULT_LLM_MODEL,
            installed_models=[],
        )
    return LlmHealthResponse(**data)


@app.post("/llm/generate", response_model=LlmGenerateResponse)
def llm_generate(
    payload: LlmGenerateRequest,
    ollama_client: OllamaClient = Depends(get_ollama_client),
) -> LlmGenerateResponse:
    try:
        data = ollama_client.generate(
            prompt=payload.prompt,
            model=payload.model,
            system=payload.system,
            temperature=payload.temperature,
        )
    except OllamaUnavailableError as exc:
        raise HTTPException(status_code=503, detail=f"Ollama is unavailable: {exc}") from exc
    return LlmGenerateResponse(**data)


@app.post("/agent/answer", response_model=AgentResponse)
def agent_answer(
    payload: AgentRequest,
    agent_service: StudyAgentService = Depends(get_agent_service),
) -> AgentResponse:
    try:
        data = agent_service.answer(
            question=payload.question,
            model=payload.model,
            temperature=payload.temperature,
        )
    except OllamaUnavailableError as exc:
        raise HTTPException(status_code=503, detail=f"Ollama is unavailable: {exc}") from exc
    return AgentResponse(**data)
