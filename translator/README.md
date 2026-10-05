# 🌐 Local Speech Translator

Полностью локальный переводчик речи без облаков и API-ключей.

**Стек:**
- 🎙️ **Whisper** (faster-whisper) — распознавание речи + автоопределение языка
- 🔤 **LibreTranslate** — перевод на английский
- 🐳 **Docker Compose** — всё запускается одной командой

---

## Быстрый старт

### 1. Требования
- Docker + Docker Compose
- 2–4 ГБ свободной оперативной памяти
- ~2 ГБ места на диске (модели скачаются автоматически)

### 2. Запуск

```bash
docker compose up --build
```

Первый запуск займёт несколько минут — скачаются модели Whisper и языковые пакеты LibreTranslate.

### 3. Открыть

Перейди в браузере на:
```
http://localhost:3000
```

Нажми **Check** чтобы убедиться что бэкенд работает, затем жми 🎙️ и говори.

---

## Конфигурация

### Изменить модель Whisper

В `docker-compose.yml` найди:
```yaml
environment:
  - WHISPER_MODEL=base
```

Доступные модели (чем больше — тем точнее, но медленнее):

| Модель   | Размер | VRAM  | Скорость  |
|----------|--------|-------|-----------|
| `tiny`   | 75 MB  | ~1 GB | Очень быстро |
| `base`   | 145 MB | ~1 GB | Быстро ✅ |
| `small`  | 466 MB | ~2 GB | Хорошо    |
| `medium` | 1.5 GB | ~5 GB | Отлично   |
| `large-v3` | 3 GB | ~10 GB | Максимум |

### Добавить языки для перевода

В `docker-compose.yml`:
```yaml
- LT_LOAD_ONLY=en,ru,zh,fr,de,es,ar,ja,ko,pt,hi,tr,vi,uk,pl,it
```
Добавь нужный ISO-код. Полный список: https://libretranslate.com/languages

### GPU (намного быстрее)

В `backend/Dockerfile` замени:
```
FROM python:3.11-slim
```
на:
```
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
```

И в `main.py`:
```python
model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
```

---

## Архитектура

```
Браузер (localhost:3000)
    │
    │  POST /transcribe (audio/webm)
    ▼
backend:8000  ← faster-whisper
    │
    │  POST /translate
    ▼
libretranslate:5001
```

---

## Остановка

```bash
docker compose down
```

Чтобы удалить скачанные модели:
```bash
docker compose down -v
```
