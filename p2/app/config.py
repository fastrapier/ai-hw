from __future__ import annotations

import os
from pathlib import Path


P2_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = P2_DIR.parent

DEFAULT_SENTIMENT_MODEL = os.environ.get(
    "SENTIMENT_MODEL",
    "cardiffnlp/twitter-xlm-roberta-base-sentiment",
)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT_SECONDS = float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "120"))
DEFAULT_LLM_MODEL = os.environ.get("DEFAULT_LLM_MODEL", "llama3.2:3b")
LANGFUSE_BASE_URL = os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST")
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY")
LANGFUSE_TRACE_NAME_PREFIX = os.environ.get("LANGFUSE_TRACE_NAME_PREFIX", "p2")

REPORT_PATH = P2_DIR / "report.md"
README_PATH = P2_DIR / "README.md"
EVAL_RESULTS_PATH = P2_DIR / "eval" / "results.md"
TASK_PATH = P2_DIR / "task.md"
