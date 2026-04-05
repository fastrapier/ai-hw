# Текст: тональность (Hugging Face)

## ТЗ (кратко)

**Задача:** по произвольному тексту оценить тональность (positive / negative / neutral в зависимости от головы модели).  
**Вход:** один или несколько файлов `*.txt` в каталоге `input/`.  
**Выход:** `output/result.json` — метки и уверенность.  
**Ограничения:** CPU в контейнере; модель мультиязычная, до 512 токенов с усечением.

**Будущий API:** удобно отдать `POST /predict` с телом `{"text": "..."}` и ответом как в `result.json`.

## Модель

`cardiffnlp/twitter-xlm-roberta-base-sentiment` — fine-tuned XLM-RoBERTa для sentiment на множестве языков.

## Сборка и запуск

Из каталога `p1/`:

```bash
docker compose build text
docker compose run --rm text
```

Или вручную:

```bash
docker build --platform linux/arm64 -t p1-text ./text
mkdir -p text/input text/output
echo "Отличный сервис, всем рекомендую." > text/input/sample.txt
docker run --rm --platform linux/arm64 \
  -v "$(pwd)/text/input:/workspace/input:ro" \
  -v "$(pwd)/text/output:/workspace/output" \
  -v hf_cache:/root/.cache/huggingface \
  p1-text
```

Переменные окружения: `INPUT_DIR`, `OUTPUT_DIR`, `SENTIMENT_MODEL`.
