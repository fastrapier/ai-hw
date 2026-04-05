# Видео: детекция объектов по кадрам (Hugging Face DETR)

## ТЗ (кратко)

**Задача:** из видеофайла взять до **N** кадров с шагом `FRAME_STRIDE` и для каждого запустить **DETR** (COCO).  
**Вход:** первый файл `.mp4`, `.avi`, `.mov`, `.mkv` в `input/`. Если нет — генерируется короткий синтетический `.avi` (OpenCV), чтобы пайплайн был воспроизводим без медиа.  
**Выход:** `output/result.json` — список кадров и боксы/классы.  
**Ограничения:** CPU; первый запуск качает веса DETR в общий volume `hf_cache`.

Переменные окружения: `MAX_FRAMES` (по умолчанию 3), `FRAME_STRIDE` (10), `SCORE_THRESHOLD` (0.5), `DETR_MODEL`.

## Модель

`facebook/detr-resnet-50` — детекция в стиле Transformer, обучена на COCO.

## Сборка и запуск

```bash
cd p1
docker compose build video
docker compose run --rm video
```

Положите клип в `video/input/`.
