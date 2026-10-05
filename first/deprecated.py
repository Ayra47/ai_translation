# МОДЕЛЬ АЖ 16ГБ ЕБАТЬ У МЕНЯ ПАМЯТИ НЕТ
class EraXTranslator:
    def __init__(self, target_lang="English"):
        if not transformers_available:
            raise ImportError("transformers и torch не установлены. pip install transformers torch")
        
        model_path = "erax-ai/EraX-Translator-V1.0"
        print(f"Загрузка переводчика EraX ({model_path})...")
        
        # Определяем устройство
        if torch.cuda.is_available():
            self.device = "cuda"
            dtype = torch.bfloat16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            # dtype = torch.float16
            dtype = torch.float32 # 
        else:
            self.device = "cpu"
            dtype = torch.float32
        
        print(f"Устройство: {self.device}")
        
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            # device_map="auto"
            device_map="mps",        # явно указываем mps вместо "auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.target_lang = target_lang
        
        self.system_prompt = """Bạn là Trợ lý AI xuất sắc về dịch thuật nhiều ngôn ngữ, đặc biệt tiếng Anh, tiếng Trung Hoa, tiếng Việt. 
Bạn cũng là 1 Hoà thượng Phật giáo uyên thâm về dịch thuật Cổ văn Trung Quốc. Người dùng sẽ giao nhiệm vụ dịch thuật cho bạn từ ngôn ngữ bất kỳ sang một ngôn ngữ được chỉ định.
Nhiệm vụ của bạn là dịch thật sát nghĩa, thể hiện đúng ý của bài gốc và không chế tác hay bịa đặt gì thêm. Không được bình luận, không được cung cấp lời giới thiệu hay mở bài hay kết luận gì, chỉ dịch thật sát nghĩa."""
        
        print("Переводчик готов!\n")
    
    def translate(self, text):
        """Переводит текст и возвращает результат."""
        instruct = f"\nDịch sang tiếng {self.target_lang}."
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text + instruct}
        ]
        
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            # outputs = self.model.generate(
            #     **inputs,
            #     max_new_tokens=512,
            #     temperature=0.2,
            #     top_k=64,
            #     top_p=0.95,
            #     repetition_penalty=1.05,
            #     # do_sample=True,
            #     do_sample=False if self.device == "mps" else True,  # ← фикс
            # )
            # bcs i have MACOS
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                repetition_penalty=1.05,
                do_sample=False,
            )

        
        # Декодируем только новые токены
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return result.strip()
    
    def translate_async(self, text, callback):
        """Переводит в отдельном потоке, вызывает callback(result) по готовности."""
        def run():
            try:
                result = self.translate(text)
                callback(result)
            except Exception as e:
                callback(f"[Ошибка перевода: {e}]")
        
        t = Thread(target=run, daemon=True)
        t.start()
