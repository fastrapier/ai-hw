# Локальная LLM (Ollama в Docker)

## ТЗ (кратко)

**Задача:** запустить **опенсорсную** LLM **локально** (на вашей машине), без облачных API провайдеров.  
**Реализация:** сервис `ollama` в `[docker-compose.yml](../docker-compose.yml)` (образ `ollama/ollama`, платформа `linux/arm64`).  
**Рекомендуемый размер:** 3B–8B; для CPU/M4 в Docker удобнее небольшие квантованные варианты.

## Запуск

Из каталога `p1/`:

```bash
docker compose up -d ollama
```

Подождать готовность сервера (несколько секунд), затем скачать модель (пример — ~3B):

```bash
docker compose exec ollama ollama pull llama3.2:3b
```

Проверка (генерация в контейнере):

```bash
docker compose exec ollama ollama run llama3.2:3b "Кратко опиши, что такое машинное обучение."
```

С хоста (если порт `11434` проброшен):

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2:3b",
  "prompt": "Привет! Одним предложением: что такое Docker?",
  "stream": false
}'
```

## Данные и кэш

Модели хранятся в Docker volume `ollama_data` (см. compose). Образы и веса **не** засоряют системный Python на macOS.

## Примечания

- Другие модели: `ollama pull qwen2.5:3b`, `ollama pull phi3:mini` и т.д. — см. [ollama.com/library](https://ollama.com/library).

