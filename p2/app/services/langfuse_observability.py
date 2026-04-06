from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from p2.app.config import (
    LANGFUSE_BASE_URL,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_TRACE_NAME_PREFIX,
)

logger = logging.getLogger(__name__)


class LangfuseObservability:
    def __init__(
        self,
        base_url: str | None = LANGFUSE_BASE_URL,
        public_key: str | None = LANGFUSE_PUBLIC_KEY,
        secret_key: str | None = LANGFUSE_SECRET_KEY,
        trace_name_prefix: str = LANGFUSE_TRACE_NAME_PREFIX,
    ) -> None:
        self.base_url = base_url
        self.public_key = public_key
        self.secret_key = secret_key
        self.trace_name_prefix = trace_name_prefix
        self._client = None

        if not (base_url and public_key and secret_key):
            return

        from langfuse import Langfuse

        self._client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            base_url=base_url,
        )

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def shutdown(self) -> None:
        if self._client is None:
            return
        self._client.shutdown()

    def trace_agent_answer(
        self,
        *,
        question: str,
        selected_tools: list[str],
        context_sources: list[str],
        prompt: str,
        system_prompt: str,
        requested_model: str | None,
        effective_model: str,
        temperature: float,
        generate: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        logger.info(
            "agent_answer_started",
            extra={
                "selected_tools": selected_tools,
                "context_sources": context_sources,
                "effective_model": effective_model,
            },
        )
        if self._client is None:
            result = generate()
            logger.info("agent_answer_completed_without_langfuse")
            return result

        try:
            with self._client.start_as_current_observation(
                name="agent-answer",
                as_type="agent",
                input={"question": question, "temperature": temperature},
                metadata={
                    "trace_name": f"{self.trace_name_prefix}-agent-answer",
                    "selected_tools": selected_tools,
                    "context_sources": context_sources,
                    "requested_model": requested_model,
                    "effective_model": effective_model,
                    "tags": ["p2", "agent", "langfuse"],
                },
            ) as agent_observation:
                with agent_observation.start_as_current_observation(
                    name="context-selection",
                    as_type="tool",
                    input={"question": question},
                    metadata={
                        "selected_tools": selected_tools,
                        "context_sources": context_sources,
                    },
                ) as context_observation:
                    context_observation.update(
                        output={
                            "selected_tools": selected_tools,
                            "context_sources": context_sources,
                        }
                    )

                with agent_observation.start_as_current_observation(
                    name="llm-generation",
                    as_type="generation",
                    model=effective_model,
                    input={
                        "system_prompt": system_prompt,
                        "prompt": prompt,
                    },
                    metadata={"requested_model": requested_model},
                    model_parameters={"temperature": temperature},
                ) as generation_observation:
                    result = generate()
                    usage_details = None
                    if isinstance(result.get("eval_count"), int):
                        usage_details = {"completion_tokens": result["eval_count"]}
                    generation_observation.update(
                        output=result.get("response", ""),
                        usage_details=usage_details,
                    )

                agent_observation.update(
                    output={
                        "answer": result.get("response", ""),
                        "selected_tools": selected_tools,
                        "context_sources": context_sources,
                        "model": result.get("model", effective_model),
                    }
                )

            logger.info("agent_answer_completed_with_langfuse")
            return result
        except Exception:
            logger.exception("langfuse_instrumentation_failed")
            return generate()
