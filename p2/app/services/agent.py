from __future__ import annotations

from pathlib import Path

from p2.app.config import EVAL_RESULTS_PATH, README_PATH, REPORT_PATH, REPO_ROOT, TASK_PATH
from p2.app.services.langfuse_observability import LangfuseObservability
from p2.app.services.ollama import OllamaClient


class StudyAgentService:
    def __init__(
        self,
        ollama_client: OllamaClient,
        observability: LangfuseObservability | None = None,
    ) -> None:
        self.ollama_client = ollama_client
        self.observability = observability

    def _read_text(self, path: Path) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8").strip()

    def _tool_project_report(self) -> tuple[str, str]:
        return ("project_report", self._read_text(REPORT_PATH))

    def _tool_api_readme(self) -> tuple[str, str]:
        return ("api_readme", self._read_text(README_PATH))

    def _tool_llm_eval(self) -> tuple[str, str]:
        return ("llm_evaluation", self._read_text(EVAL_RESULTS_PATH))

    def _tool_assignment(self) -> tuple[str, str]:
        return ("assignment_brief", self._read_text(TASK_PATH))

    def _select_tools(self, question: str) -> list[tuple[str, str]]:
        lowered = question.lower()
        tools = [self._tool_project_report()]

        if any(keyword in lowered for keyword in ("api", "fastapi", "endpoint", "predict", "ручк")):
            tools.append(self._tool_api_readme())
        if any(keyword in lowered for keyword in ("llm", "модель", "агент", "галлюцина", "ollama")):
            tools.append(self._tool_llm_eval())
        if any(keyword in lowered for keyword in ("задани", "тз", "требован", "част")):
            tools.append(self._tool_assignment())

        seen: set[str] = set()
        unique_tools: list[tuple[str, str]] = []
        for name, content in tools:
            if name in seen or not content:
                continue
            seen.add(name)
            unique_tools.append((name, content))
        return unique_tools

    def answer(self, question: str, model: str | None = None, temperature: float = 0.2) -> dict[str, object]:
        selected_tools = self._select_tools(question)
        selected_tool_names = [name for name, _ in selected_tools]
        context_sources = self._context_sources(selected_tools)
        context_blocks = []
        for name, content in selected_tools:
            trimmed = content[:3000]
            context_blocks.append(f"[{name}]\n{trimmed}")

        system_prompt = (
            "You are a study assistant for the p2 project. "
            "Answer only from the provided project context. "
            "If the answer is missing, say that it is not documented yet."
        )
        prompt = "\n\n".join(
            [
                "Project context:",
                "\n\n".join(context_blocks),
                f"Question: {question}",
                "Answer in Russian with short, concrete paragraphs.",
            ]
        )
        effective_model = model or self.ollama_client.default_model

        if self.observability is not None:
            result = self.observability.trace_agent_answer(
                question=question,
                selected_tools=selected_tool_names,
                context_sources=context_sources,
                prompt=prompt,
                system_prompt=system_prompt,
                requested_model=model,
                effective_model=effective_model,
                temperature=temperature,
                generate=lambda: self.ollama_client.generate(
                    prompt=prompt,
                    model=model,
                    system=system_prompt,
                    temperature=temperature,
                ),
            )
        else:
            result = self.ollama_client.generate(
                prompt=prompt,
                model=model,
                system=system_prompt,
                temperature=temperature,
            )
        return {
            "model": result["model"],
            "answer": result["response"],
            "selected_tools": selected_tool_names,
            "context_sources": context_sources,
        }

    def _context_sources(self, selected_tools: list[tuple[str, str]]) -> list[str]:
        mapping = {
            "project_report": str(REPORT_PATH.relative_to(REPO_ROOT)),
            "api_readme": str(README_PATH.relative_to(REPO_ROOT)),
            "llm_evaluation": str(EVAL_RESULTS_PATH.relative_to(REPO_ROOT)),
            "assignment_brief": str(TASK_PATH.relative_to(REPO_ROOT)),
        }
        return [mapping[name] for name, _ in selected_tools if name in mapping]
