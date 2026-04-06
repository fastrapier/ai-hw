COMPOSE := docker compose -f docker-compose.yml

.PHONY: help build models api up test eval ps logs health llm-health down clean

help:
	@echo "Available targets:"
	@echo "  make build       - build Docker images"
	@echo "  make models      - start Ollama and pull required models"
	@echo "  make api         - start API container"
	@echo "  make up          - full demo setup: build, models, tests, api"
	@echo "  make test        - run pytest in Docker"
	@echo "  make eval        - run LLM evaluation in Docker"
	@echo "  make ps          - show compose services"
	@echo "  make logs        - tail API and Ollama logs"
	@echo "  make health      - call GET /health"
	@echo "  make llm-health  - call GET /llm/health"
	@echo "  make down        - stop containers"
	@echo "  make clean       - stop containers and remove volumes"

build:
	$(COMPOSE) build

models:
	$(COMPOSE) up -d ollama
	$(COMPOSE) run --rm ollama-init

api:
	$(COMPOSE) up -d api

up: build models test api
	@echo "p2 is ready on http://127.0.0.1:8000"

test:
	$(COMPOSE) run --rm tests

eval:
	$(COMPOSE) run --rm eval

ps:
	$(COMPOSE) ps

logs:
	$(COMPOSE) logs -f api ollama

health:
	curl -s http://127.0.0.1:8000/health
	@echo

llm-health:
	curl -s http://127.0.0.1:8000/llm/health
	@echo

down:
	$(COMPOSE) down

clean:
	$(COMPOSE) down -v
