COMPOSE := docker compose -f docker-compose.yml

.PHONY: help build observability models api up test ps logs health agent-demo down clean

help:
	@echo "Available targets:"
	@echo "  make build         - build Docker images for the p3 demo"
	@echo "  make observability - start Langfuse and its dependencies"
	@echo "  make models        - start Ollama and pull required model"
	@echo "  make api           - start the instrumented API"
	@echo "  make up            - full demo setup: build, observability, model, tests, api"
	@echo "  make test          - run existing p2 API tests in Docker"
	@echo "  make ps            - show compose services"
	@echo "  make logs          - tail API and Langfuse logs"
	@echo "  make health        - call API and Langfuse health endpoints"
	@echo "  make agent-demo    - send a demo request to /agent/answer"
	@echo "  make down          - stop containers"
	@echo "  make clean         - stop containers and remove volumes"

build:
	$(COMPOSE) build

observability:
	$(COMPOSE) up -d postgres clickhouse redis minio langfuse-worker langfuse-web

models:
	$(COMPOSE) up -d ollama
	$(COMPOSE) run --rm ollama-init

api:
	$(COMPOSE) up -d api

up: build observability models test api
	@echo "p3 demo is ready: API on http://127.0.0.1:8001, Langfuse on http://127.0.0.1:3001"

test:
	$(COMPOSE) run --rm tests

ps:
	$(COMPOSE) ps

logs:
	$(COMPOSE) logs -f api langfuse-web langfuse-worker

health:
	curl -s http://127.0.0.1:8001/health
	@echo
	curl -s http://127.0.0.1:8001/llm/health
	@echo

agent-demo:
	curl -s http://127.0.0.1:8001/agent/answer \
		-H "Content-Type: application/json" \
		-d '{"question":"Какая модель выбрана для агента и почему?"}'
	@echo

down:
	$(COMPOSE) down

clean:
	$(COMPOSE) down -v
