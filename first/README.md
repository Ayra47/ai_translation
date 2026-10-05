# Описание проекта

1. **Получение текста из голоса через Vosk**
   - Поддержка: английский, русский, вьетнамский
   - В будущем: Faster-Whisper

2. **Перевод текста на другой язык**
   - Используется EraX-Translator-V1.0 (легковесная модель 4B на базе Gemma3-4B)
   - Автоматический перевод транскрибированного текста
   - Поддержка 10 языков перевода: en, ru, vi, zh, ja, ko, fr, de, es, it

3. **TTS (в разработке)**
   - Mistral Voxtral TTS — лучший универсальный выбор (9 языков, НЕТ русского и вьетнамского)
   - NVIDIA Magpie TTS Multilingual — вьетнамский + 8 языков
   - Supertone Sona2 (платное) — 23 языка включая русский и вьетнамский
   - Qwen3-TTS-12Hz-0.6B-CustomVoice — быстрый, 10 языков

# Установка

```bash
# Create a virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

# Запуск

## Базовое использование

```bash
# Транскрибация с микрофона (русский + английский)
python vosk_transcriber.py -l ru,en --mic -o text.txt

# Транскрибация из файла
python vosk_transcriber.py -l en -i audio.wav
```

## С переводом

```bash
# запуск переводчика translategemma
ollama run translategemma

# Русская речь → английский текст
python vosk_transcriber.py -l ru --mic --translate en

# Вьетнамская речь → русский текст
python vosk_transcriber.py -l vi --mic --translate ru

# Мультиязычная речь → перевод на английский
python vosk_transcriber.py -l ru,vi,en --mic --translate en
```

## Параметры

```
-l, --language          Язык(и) распознавания (en, ru, vi)
--mic                   Использовать микрофон
-i, --input             Входной WAV файл
--translate             Перевод на целевой язык (en, ru, vi, zh, ja, ko, fr, de, es, it)
-o, --output            Сохранить результат в файл
--raw                   Только текст без тегов языка
--confidence            Порог уверенности (по умолчанию 0.1)
--list-languages        Показать все поддерживаемые языки
--list-models           Показать доступные модели
```

# Примеры

```bash
# Распознавание + перевод на русский
python vosk_transcriber.py -l en --mic --translate ru

# Распознавание + перевод на английский
python vosk_transcriber.py -l ru --mic --translate en

# Сохранение с переводом
python vosk_transcriber.py -l ru,vi --mic --translate en -o result.txt
```

# Деактивация venv

```bash
deactivate
```