import os
import json
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


def load_dialogue_history(file_path: str) -> List[Dict[str, str]]:
    if os.path.exists(file_path):
        # если файл пустой, сразу возвращаем []
        if os.path.getsize(file_path) == 0:
            return []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            # файл повреждён или не JSON — начинаем с чистой истории
            return []
    return []



def history_to_text(history: List[Dict[str, str]]):
    """Преобразует историю диалога в читаемый текстовый формат"""
    lines = []
    for entry in history:
        role = entry['role']
        text = entry['text']
        lines.append(f"[{role}] {text}")
    return "\n".join(lines)


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

        # Путь к файлу истории диалога
        self.history_file = os.getenv("DIALOGUE_HISTORY_FILE", "dialogue_history.json")
        # Загрузка или инициализация истории диалога
        self.dialogue_history: List[Dict[str, str]] = load_dialogue_history(self.history_file)
        if not self.dialogue_history:
            # Добавляем стартовое приветствие оператора, если файл пустой
            self.add_to_history(role="operator", text="Здравствуйте! Чем могу помочь?")

        # Шаблон для определения намерения
        self.intent_prompt = PromptTemplate(
            template="""
Привет. Ты ассистент консультанта call-центра. Тебе нужно определить основное намерение клиента 
на основании запроса клиента и истории диалога консультанта с клиентом :{history}.

Извлекай только суть проблемы, без лишних слов.

Запрос пользователя: {user_input}

В ответе верни только намерение.""",
            input_variables=["history", "user_input"]
        )

        # Шаблон для генерации рекомендаций оператору
        self.action_suggestion_prompt = PromptTemplate(
            template="""У тебя есть история диалога:
{history}

На основе следующей информации сгенерируй рекомендации для оператора поддержки:
1. Намерение пользователя: {intent}
2. Эмоциональное состояние пользователя: {emotion}
3. Найденное решение из базы знаний: {solution_content}
4. Уверенность в решении (score): {similarity_score}

Учти эмоциональное состояние пользователя при формулировке рекомендаций.
Если уверенность в решении низкая (score < 0.7), предложи уточняющие вопросы.
Предложи оптимальный способ коммуникации с пользователем, учитывая его эмоции.
Добавь любые дополнительные рекомендации, которые могут помочь оператору.

В ответе верни JSON-массив объектов:
[
  {{
    "hint": "текст подсказки для оператора",
    "reasoning": "как пришли к подсказке"
  }}
]""",
            input_variables=["history", "intent", "emotion", "solution_content", "similarity_score"]
        )

    def add_to_history(self, role: str, text: str) -> None:
        """Добавляет запись в историю диалога и сохраняет в файл, избегая дублирования"""
        entry = {"role": role, "text": text}
        # Проверка на дублирование: не добавлять, если последнее сообщение совпадает
        if not self.dialogue_history or self.dialogue_history[-1] != entry:
            self.dialogue_history.append(entry)
            # Сохраняем историю в JSON-файл
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.dialogue_history, f, ensure_ascii=False, indent=2)

    def speech_to_text(self, audio_path: str) -> str:
        """Заглушка для ASR системы (реализуйте с Whisper или другим ASR)"""
        return "У меня телефон samsung, операционная система android"

    def get_intent(self, text: str) -> str:
        """Определение намерения пользователя"""
        history_text = history_to_text(self.dialogue_history)
        chain = LLMChain(llm=self.llm, prompt=self.intent_prompt)
        result = chain.run(history=history_text, user_input=text)
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
        embedding = self.embedding_model.encode(query)
        hits = self.qdrant_client.search(
            collection_name="support_intents_labels",
            query_vector=embedding.tolist(),
            limit=1
        )
        if not hits:
            return {"name": "Не найдено", "content": "Решение не найдено в базе знаний", "score": 0.0}
        return {"name": hits[0].payload["name"], "content": hits[0].payload["content"], "score": hits[0].score}

    def generate_action_suggestions(self, intent: str, emotion: str, solution_content: str, similarity_score: float) -> str:
        """Генерация рекомендаций для оператора"""
        history_text = history_to_text(self.dialogue_history)
        chain = LLMChain(llm=self.llm, prompt=self.action_suggestion_prompt)
        result = chain.run(
            history=history_text,
            intent=intent,
            emotion=emotion,
            solution_content=solution_content,
            similarity_score=similarity_score
        )
        return str(result).strip()

    def execute(self, query: str) -> Dict[str, Any]:
        """Основной метод обработки аудио"""
        # 1. Преобразование аудио в текст
        text = query
        # 2. Сохраняем реплику клиента
        self.add_to_history(role="client", text=text)
        # 3. Определение намерения
        intent = self.get_intent(text)
        # 4. Определение эмоции
        emotion = self.get_emotion(text)
        # 5. Поиск решения
        solution = self.search_solution(intent)
        # 6. Генерация рекомендаций
        raw_suggestions = self.generate_action_suggestions(
            intent=intent,
            emotion=emotion,
            solution_content=solution["content"],
            similarity_score=solution["score"]
        )
        action_suggestions = JsonOutputParser().parse(raw_suggestions)[0]
        # 7. Сохраняем подсказку оператора
        self.add_to_history(role="hint", text=action_suggestions["hint"])
        # Возвращаем результаты и историю
        return {
            "user_input": text,
            "intent": intent,
            "emotion": emotion,
            "solution_name": solution["name"],
            "solution_content": solution["content"],
            "similarity_score": solution["score"],
            "hint": action_suggestions["hint"],
            "hint_reason": action_suggestions["reasoning"],
            "history": self.dialogue_history
        }

import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from langchain_together import ChatTogether
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class CorrectiveActionAgent:
    """
    Agent that tracks dialogue history, finds the violated rule based on consultant's text,
    and generates a corrective recommendation via ChatTogether.
    """
    def __init__(
        self,
        qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key: str = os.getenv("QDRANT_API_KEY", None),
        embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3"),
        qa_collection: str = os.getenv("QA_COLLECTION_NAME", "qa_rules"),
        llm_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        together_api_key: str = os.getenv("TOGETHER_API_KEY"),
    ):
        # History setup
        self.history_file = os.getenv("DIALOGUE_HISTORY_FILE", "dialogue_history.json")
        self.dialogue_history: List[Dict[str, str]] = load_dialogue_history(self.history_file)


        # Qdrant client
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, prefer_grpc=False)
        self.collection = qa_collection
        # Embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        # LLM via ChatTogether
        self.llm = ChatTogether(model=llm_model, api_key=together_api_key)
        # Prompt template for corrective action, now including history
        self.prompt = PromptTemplate(
            template="""
У вас есть история диалога:
{history}

Вы — наставник оператора контакт‑центра.
Нарушенное правило: "{rule_text}"
Дайте краткую рекомендацию (1–2 предложения), как оператор может скорректировать поведение в будущем.
""",
            input_variables=["history", "rule_text"]
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def add_to_history(self, role: str, text: str) -> None:
        """Добавляет запись в историю диалога и сохраняет в файл, избегая дублирования"""
        entry = {"role": role, "text": text}
        if not self.dialogue_history or self.dialogue_history[-1] != entry:
            self.dialogue_history.append(entry)
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.dialogue_history, f, ensure_ascii=False, indent=2)

    def suggest_correction(self, consultant_text: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """
        1) Сохраняет текст консультанта в историю
        2) Находит наиболее похожее правило
        3) Генерирует рекомендацию с учётом истории
        4) Сохраняет рекомендацию в историю
        """
        # 1. Логируем сообщение консультанта
        self.add_to_history(role="consultant", text=consultant_text)

        # 2. Embed consultant's text и поиск похожих правил
        vector = self.embedding_model.encode(consultant_text, convert_to_numpy=True).tolist()
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=top_k,
            with_payload=True
        )

        results = []
        history_text = history_to_text(self.dialogue_history)
        for hit in hits:
            payload = hit.payload if hasattr(hit, 'payload') else hit['payload']
            rule_text = payload.get('text') if isinstance(payload, dict) else getattr(payload, 'text')

            # 3. Генерация рекомендации с учётом истории
            suggestion = self.chain.run(history=history_text, rule_text=rule_text).strip()


            results.append({
                'rule_id': hit.id,
                'rule_text': rule_text,
                'suggestion': suggestion
            })

        return results



if __name__ == "__main__":
    pipeline = AudioProcessingPipeline()
    result = pipeline.process_audio("path/to/audio.wav")
    print(json.dumps(result, ensure_ascii=False, indent=2))
