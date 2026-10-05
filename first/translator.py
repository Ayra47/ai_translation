"""
Translator using Ollama API
"""
import queue
from threading import Thread
import urllib.request
import json


class EraXTranslator:
    """Переводчик на основе Ollama API."""
    
    def __init__(self, target_lang="English", model="translategemma:latest"):
        self.model = model
        self.target_lang = target_lang
        self.ollama_url = "http://localhost:11434/api/generate"

        # Проверяем доступность Ollama
        try:
            self._check_model()
        except Exception as e:
            print(f"Ошибка подключения к Ollama: {e}\n")
            raise

        # Очередь для последовательной обработки
        self._queue = queue.Queue()
        self._running = True
        self._worker = Thread(target=self._process_queue, daemon=False)
        self._worker.start()

        print("Переводчик готов!\n")

    def _check_model(self):
        """Проверяет, доступна ли модель в Ollama."""
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags")
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                models = [m["name"] for m in data.get("models", [])]
                if self.model not in models:
                    print(f"Модель '{self.model}' не найдена в Ollama.")
                    print(f"Доступные модели: {', '.join(models)}")
                    print(f"Запустите: ollama run {self.model}")
        except Exception as e:
            print(f"Не удалось проверить модели Ollama: {e}")

    def _process_queue(self):
        """Воркер — обрабатывает переводы по одному."""
        while self._running:
            try:
                text, callback = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                result = self.translate(text)
                callback(result)
            except Exception as e:
                callback(f"[Ошибка: {e}]")
            finally:
                self._queue.task_done()

    def stop(self):
        """Остановить воркер и дождаться завершения всех задач."""
        self._running = False
        self._queue.join()

    def translate(self, text):
        """Переводит текст через Ollama API."""
        prompt = (
            f"You are an excellent AI assistant for multilingual translation, especially English, Chinese, and Vietnamese. "
            f"Your task is to translate accurately, conveying the true meaning of the original text without adding or fabricating anything. "
            f"You are not allowed to comment, provide introductions, or conclusions; just translate as accurately as possible. "
            f"Translate to {self.target_lang}: {text}"
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 128
            }
        }
        
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            self.ollama_url,
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode())
            translation = result.get("response", "").strip()
            return translation

    def translate_async(self, text, callback):
        """Добавляет задачу в очередь."""
        self._queue.put((text, callback))
