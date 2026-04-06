# p3: LLMOps (Langfuse) for the p2 agent

`p3` covers only `Сз_3. Part 1. LLMOps (Observability)` from `p3/task.md`.

`RecSys` is intentionally out of scope for this iteration.

## What is being observed

The observed system is the agent from `p2`, available via `POST /agent/answer`.

The request flow is:

1. user sends a question;
2. `StudyAgentService` selects project context;
3. the service builds a system prompt and final prompt;
4. Ollama generates the answer;
5. the API returns `answer`, `selected_tools`, and `context_sources`.

For `p3`, this flow is instrumented with `Langfuse`.

## Why Langfuse

Three options were compared:

- `Langfuse` - purpose-built for LLM apps, gives traces, generations, prompt/response visibility, metadata, and a convenient UI for demos.
- `OpenLLMetry` / `OpenTelemetry` - flexible and standards-based, but needs more manual assembly to get a polished LLM-specific view.
- `Prometheus` + `Loki` + `Jaeger` - strong generic observability stack, but less convenient for prompt-level LLM debugging in a short course project.

The main choice criteria were:

- easy local Docker setup;
- direct applicability to an LLM agent;
- visibility into prompt, output, model, and context metadata;
- suitability for a short live demo.

## What is captured

The Langfuse instrumentation around `POST /agent/answer` captures:

- the incoming user question;
- selected context sources (`selected_tools`, `context_sources`);
- requested and effective LLM model;
- prompt and system prompt;
- generation output;
- success/error flow of the request.

The trace is split into observations such as:

- `agent-answer`
- `context-selection`
- `llm-generation`

## Runtime

`p3/docker-compose.yml` starts:

- the existing API from `p2`, now with Langfuse credentials;
- local Ollama for the agent model;
- self-hosted Langfuse Web + Worker;
- supporting services: PostgreSQL, ClickHouse, Redis, MinIO.

Ports used for the demo:

- API: `http://127.0.0.1:8001`
- Langfuse UI: `http://127.0.0.1:3001`

## Quick start

From `p3/`:

```bash
make up
```

This does the following:

- builds the Docker image reused from `p2`;
- starts Langfuse and its dependencies;
- starts Ollama and pulls `llama3.2:3b`;
- runs the existing API tests;
- starts the instrumented API.

Useful commands:

```bash
make help
make ps
make logs
make health
make agent-demo
make down
make clean
```

## Demo script

1. Run `make up`.
2. Open Langfuse UI at [http://127.0.0.1:3001](http://127.0.0.1:3001).
3. Trigger a request:

```bash
make agent-demo
```

4. In the Langfuse UI, inspect the trace and verify:

- root observation `agent-answer`;
- child observation `context-selection`;
- child observation `llm-generation`;
- prompt and output data;
- selected context metadata and model name.

## Recommended demo questions

- `Какая модель выбрана для агента и почему?`
- `Какие уязвимости есть у агентного решения?`
- `Какие endpoint'ы есть в FastAPI API проекта?`

## Failure scenario for observability

For a debugging demo, you can intentionally stop `ollama` and then call `POST /agent/answer`.

Expected outcome:

- API returns an error;
- Langfuse trace shows where the request failed;
- container logs help correlate the failure with the generation step.

## Files

- `p3/task.md` - assignment text
- `p3/README.md` - runbook for the LLMOps part
- `p3/report.md` - report for Part 1
- `p3/docker-compose.yml` - local Langfuse + API stack
- `p3/Makefile` - short demo commands
