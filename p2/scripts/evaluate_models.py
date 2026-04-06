from __future__ import annotations

import argparse
import json
from pathlib import Path

from p2.app.services.ollama import OllamaClient, OllamaUnavailableError


PROMPTS_PATH = Path(__file__).resolve().parents[1] / "eval" / "prompts.json"


def load_prompts() -> list[dict[str, str]]:
    return json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate local Ollama models on a fixed prompt set.")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model names available in Ollama, for example llama3.2:3b qwen2.5:3b phi3:mini",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the evaluation prompts.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    prompts = load_prompts()
    client = OllamaClient()

    try:
        for model in args.models:
            print(f"# Model: {model}")
            for prompt in prompts:
                result = client.generate(
                    prompt=prompt["prompt"],
                    model=model,
                    system=prompt.get("system"),
                    temperature=args.temperature,
                )
                print(f"\n## Prompt: {prompt['id']}")
                print(result["response"].strip())
                print()
    except OllamaUnavailableError as exc:
        print(f"Ollama is unavailable: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
