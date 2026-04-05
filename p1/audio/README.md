# Аудио: классификация событий YAMNet (TensorFlow Hub)

## ТЗ (кратко)

**Задача:** по аудиофрагменту получить топ классов звуковых событий (AudioSet / YAMNet).  
**Вход:** все файлы `*.wav` в `input/` (по алфавиту; моно/стерео → моно). Если файлов нет — один прогон с синтетическим синусом 440 Hz.  
**Выход:** `output/result.json` — массив `results`: для каждого файла `source` и `top5` классов со скорами.

## Модель

[YAMNet](https://tfhub.dev/google/yamnet/1) — предобученная на AudioSet, 521 класс.

## Сборка и запуск

```bash
cd p1
docker compose build audio
docker compose run --rm audio
```

Положите один или несколько `*.wav` в `audio/input/`.