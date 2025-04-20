import os
from typing import Dict, Any
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from transformers import pipeline
from langchain_together import ChatTogether
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()


class AudioProcessingPipeline:
    def __init__(self):
        # Инициализация модели для определения намерений
        self.llm = ChatTogether(model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")

        # Инициализация модели для эмбеддингов
        self.embedding_model = SentenceTransformer("BAAI/bge-m3")

        # Инициализация клиента Qdrant
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY", None)
        )

        # Инициализация модели для определения эмоций
        self.emotion_model = pipeline(model="seara/rubert-tiny2-ru-go-emotions")

        # Шаблон для определения намерения
        self.intent_prompt = PromptTemplate(
            template="""Анализируй запрос пользователя и определи основное намерение.
            Извлекай только суть проблемы, без лишних слов. 

            Примеры:
            Запрос: У меня не включается компьютер
            Намерение: не включается компьютер

            Запрос: Как исправить ошибку подключения к интернету?
            Намерение: ошибка подключения к интернету

            Запрос пользователя: {user_input}

            В качестве ответа возвращай только намерение пользователя текстом""",
            input_variables=["user_input"]
        )

    def speech_to_text(self, audio_path: str) -> str:
        """Заглушка для ASR системы (реализуйте с Whisper или другим ASR)"""
        return "Как подключиться к интернету?"

    def get_intent(self, text: str) -> str:
        """Определение намерения пользователя"""
        chain = LLMChain(llm=self.llm, prompt=self.intent_prompt)
        result = chain.run(user_input=text)

        # Очистка результата
        intent = str(result).strip().lower()
        if intent.startswith('намерение:'):
            intent = intent.replace('намерение:', '').strip()
        return intent

    def get_emotion(self, text: str) -> str:
        """Определение эмоции"""
        result = self.emotion_model(text)
        return result[0]["label"]

    def search_solution(self, query: str) -> Dict[str, Any]:
        """Поиск решения в базе знаний"""
        # Создание эмбеддинга
        embedding = self.embedding_model.encode(query)

        # Поиск в Qdrant
        hits = self.qdrant_client.search(
            collection_name="support_intents",
            query_vector=embedding.tolist(),
            limit=1
        )

        if not hits:
            return {
                "name": "Не найдено",
                "content": "Решение не найдено в базе знаний",
                "score": 0.0
            }

        return {
            "name": hits[0].payload["name"],
            "content": hits[0].payload["content"],
            "score": hits[0].score
        }

    def process_audio(self, audio_path: str) -> Dict[str, Any]:
        """Основной метод обработки аудио"""
        # 1. Преобразование аудио в текст
        text = self.speech_to_text(audio_path)

        # 2. Определение намерения
        intent = self.get_intent(text)

        # 3. Определение эмоции
        emotion = self.get_emotion(text)

        # 4. Поиск решения
        solution = self.search_solution(intent)

        return {
            "user_input": text,
            "intent": intent,
            "emotion": emotion,
            "solution_name": solution["name"],
            "solution_content": solution["content"],
            "similarity_score": solution["score"]
        }


if __name__ == "__main__":
    pipeline = AudioProcessingPipeline()

    # Обработка аудио (передайте реальный путь к файлу)
    result = pipeline.process_audio("path/to/audio.wav")

    print(f"Распознанный текст: {result['user_input']}")
    print(f"Намерение пользователя: {result['intent']}")
    print(f"Эмоция пользователя: {result['emotion']}")
    print(f"Проблема из справки: {result['solution_name']}")
    print(f"Решение проблемы: {result['solution_content']}")
    print(f"Score: {result['similarity_score']}")