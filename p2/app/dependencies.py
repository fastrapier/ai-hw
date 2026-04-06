from __future__ import annotations

from fastapi import Request

from p2.app.services.agent import StudyAgentService
from p2.app.services.ollama import OllamaClient
from p2.app.services.sentiment import SentimentService


def get_sentiment_service(request: Request) -> SentimentService:
    return request.app.state.sentiment_service


def get_ollama_client(request: Request) -> OllamaClient:
    return request.app.state.ollama_client


def get_agent_service(request: Request) -> StudyAgentService:
    return request.app.state.agent_service
