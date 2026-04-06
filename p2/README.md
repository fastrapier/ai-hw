# p2: FastAPI для ML, LLM и агента

В этой части репозитория находится API для:

- текстовой модели тональности из `p1/text`;
- локальной LLM через Ollama;
- простого учебного агента для подготовки к защите проекта.

## Структура

- `app/main.py` - FastAPI приложение.
- `tests/test_api.py` - тесты API.
- `eval/prompts.json` - набор промптов для сравнения LLM.
- `eval/results.md` - результаты сравнения моделей и выбор модели.
- `scripts/evaluate_models.py` - скрипт для автоматического прогона промптов через Ollama.
- `report.md` - итоговый отчёт по частям 1-5.

## Запуск  в Docker

Из каталога `p2/` проще всего использовать `Makefile`:

```bash
make up
```

## Запуск API

Если хочется запускать шаги по отдельности:

```bash
make build
make models
make api
```

После запуска доступны маршруты:

- `GET /health`
- `POST /predict`
- `GET /llm/health`
- `POST /llm/generate`
- `POST /agent/answer`

Пример запроса к ML API:

```bash
curl http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Это очень полезный сервис"}'
```

Пример запроса к LLM API:

```bash
curl http://127.0.0.1:8000/llm/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Кратко объясни, что делает этот проект","model":"llama3.2:3b"}'
```

## Агент

В `p2` есть простой учебный агент для подготовки к защите проекта. Он отвечает на вопросы по API, выбору LLM, метрикам и требованиям задания.

Агент доступен через `POST /agent/answer`.

Перед ответом агент подбирает локальный контекст по проекту:

- `p2/report.md`
- `p2/README.md`
- `p2/eval/results.md`
- `p2/task.md`

В коде это отражено как внутренние источники контекста:

- `project_report`
- `api_readme`
- `llm_evaluation`
- `assignment_brief`

Пример запроса к агенту:

```bash
curl http://127.0.0.1:8000/agent/answer \
  -H "Content-Type: application/json" \
  -d '{"question":"Какая модель выбрана для агента и почему?"}'
```

Пример ответа:

```json
{
  "model": "llama3.2:3b",
  "answer": "Для агента выбрана llama3.2:3b, так как она показала более осторожное поведение и лучше справилась с проверкой на галлюцинации.",
  "selected_tools": ["project_report", "llm_evaluation"],
  "context_sources": ["p2/report.md", "p2/eval/results.md"]
}
```

Расшифровка полей ответа:

- `answer` - текст ответа модели;
- `selected_tools` - какие логические источники контекста агент выбрал перед генерацией;
- `context_sources` - из каких файлов был реально взят контекст.

Лучше всего агент отвечает на вопросы, связанные именно с проектом, например:

- какая модель выбрана для агента;
- какие endpoint'ы есть в API;
- какие метрики и уязвимости описаны в отчёте;
- что требуется в части 5 задания.

## Что делает compose

- `api` - FastAPI приложение на `http://127.0.0.1:8000`
- `ollama` - локальный сервер LLM внутри compose-сети `p2`
- `ollama-init` - разовая загрузка моделей `llama3.2:3b`, `qwen2.5:3b`, `phi3:mini`
- `tests` - запуск `pytest` внутри контейнера
- `eval` - прогон оценочного набора против трёх моделей

## Полный сценарий для показа

Если хочется минимизировать риски на демонстрации, можно использовать такой порядок:

```bash
cd p2
make up
```

После этого у тебя уже есть:

- поднятый API;
- загруженные модели в volume `ollama_data`;
- кэш Hugging Face в volume `hf_cache`;
- прогнанные тесты без системного Python.

По умолчанию API внутри compose обращается к `http://ollama:11434`. Порт Ollama наружу специально не публикуется, чтобы `p2` не конфликтовал с уже запущенным `p1` или другим локальным инстансом Ollama.

## Тесты

```bash
make test
```

## Оценка моделей

Скрипт оценки читает `p2/eval/prompts.json` и делает вызовы к Ollama внутри compose:

```bash
make eval
```

## Полезные команды

```bash
make help
make ps
make logs
make health
make llm-health
```

## Остановка

```bash
make down
```

Если нужно остановить контейнеры и удалить model/cache volumes:

```bash
make clean
```

