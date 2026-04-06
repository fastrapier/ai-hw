# Серия заданий 1 (с1сз1) — ПрИнж

Четыре демо по разным модальностям + локальная LLM, воспроизводимые в **Docker** (`linux/arm64`, Apple Silicon). Матрица стеков: см. [MATRIX.md](./MATRIX.md). Полное ТЗ курса: [сз1.md](./сз1.md).

## Быстрый старт

Требования: **Docker Desktop** (или совместимый engine), архитектура **arm64**.

```bash
cd p1

# ML-демо (каждый сервис отдельно)
docker compose build text audio image video
docker compose run --rm text
docker compose run --rm audio
docker compose run --rm image
docker compose run --rm video

# LLM (фоновый сервис + pull модели)
docker compose up -d ollama
docker compose exec ollama ollama pull llama3.2:3b
docker compose exec ollama ollama run llama3.2:3b "Привет!"
```

Результаты появляются в `*/output/result.json`. Входные данные кладите в `*/input/` (см. README в каждой папке).

## Структура

| Каталог | Модальность | Стек |
|---------|-------------|------|
| [text](./text/) | Текст, тональность | Hugging Face |
| [audio](./audio/) | Аудио, YAMNet | TensorFlow Hub |
| [image](./image/) | Изображение, ImageNet | PyTorch / torchvision |
| [video](./video/) | Видео, кадры + DETR | Hugging Face |
| [llm](./llm/) | Локальная LLM | Ollama |

Именованные volumes в compose: `hf_cache`, `tfhub_cache`, `torch_cache`, `ollama_data`.

## Отчёт

Сводный отчёт по заданию: [report.md](./report.md).
