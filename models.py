import json
import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
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

        self.dialogue_history: List[Dict[str, str]] = []
        self.history_file = os.getenv("DIALOGUE_HISTORY_FILE", "dialogue_history.json")

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

        # Шаблон для генерации рекомендаций оператору
        self.action_suggestion_prompt = PromptTemplate(
            template="""На основе следующей информации сгенерируй рекомендации для оператора поддержки:
            1. Намерение пользователя: {intent}
            2. Эмоциональное состояние пользователя: {emotion}
            3. Найденное решение из базы знаний: {solution_content}
            4. Уверенность в решении (score): {similarity_score}

            Учти эмоциональное состояние пользователя при формулировке рекомендаций.
            Если уверенность в решении низкая (score < 0.7), предложи уточняющие вопросы.
            Предложи оптимальный способ коммуникации с пользователем, учитывая его эмоции.
            Добавь любые дополнительные рекомендации, которые могут помочь оператору.

            В ответе верни в виже json:
            [
            {{
              "hint" : подсказка, что надо сказать консультанту для решения проблемы клиента
              "reasoning" : краткое описание того, как ты пришел к генерации подсказки в поле hint
            }}
            ]
            """,
            input_variables=["intent", "emotion", "solution_content", "similarity_score"]
        )

    def add_to_history(self, role: str, text: str) -> None:
        """Добавляет запись в историю диалога и сохраняет в файл"""
        entry = {"role": role, "text": text}
        self.dialogue_history.append(entry)
        # Сохраняем историю в JSON-файл
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(self.dialogue_history, f, ensure_ascii=False, indent=2)

    def speech_to_text(self, audio_path: str) -> str:
        """Заглушка для ASR системы (реализуйте с Whisper или другим ASR)"""
        return "Я пополнил баланс, а деньги не пришли!"

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

    def generate_action_suggestions(self, intent: str, emotion: str, solution_content: str, similarity_score: float) -> str:
        """Генерация рекомендаций для оператора"""
        chain = LLMChain(llm=self.llm, prompt=self.action_suggestion_prompt)
        result = chain.run(
            intent=intent,
            emotion=emotion,
            solution_content=solution_content,
            similarity_score=similarity_score
        )
        return str(result).strip()

    def process_audio(self, audio_path: str) -> Dict[str, Any]:
        """Основной метод обработки аудио"""
        # 1. Преобразование аудио в текст
        text = self.speech_to_text(audio_path)
        self.add_to_history(role="client", text=text)
        # 2. Определение намерения
        intent = self.get_intent(text)

        # 3. Определение эмоции
        emotion = self.get_emotion(text)

        # 4. Поиск решения
        solution = self.search_solution(intent)

        # 5. Генерация рекомендаций для оператора
        result = self.generate_action_suggestions(
            intent=intent,
            emotion=emotion,
            solution_content=solution["content"],
            similarity_score=solution["score"]
        )
        action_suggestions = JsonOutputParser().parse(result)[0]

        self.add_to_history(role="hint", text=action_suggestions["hint"])

        return {
            "user_input": text,
            "intent": intent,
            "emotion": emotion,
            "solution_name": solution["name"],
            "solution_content": solution["content"],
            "similarity_score": solution["score"],
            "hint": action_suggestions["hint"],
            "hint_reason" : action_suggestions["reasoning"]
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
    print("\nРекомендации для оператора:")
    print(result['hint'])
    print("\nАргументация:")
    print(result['hint_reason'])