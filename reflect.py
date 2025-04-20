import os
import json
from uuid import uuid4
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from langchain_together import ChatTogether
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()


def load_dialogue_history(file_path: str) -> List[Dict[str, str]]:
    """Загружает историю диалога из JSON-файла"""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def history_to_text(history: List[Dict[str, str]]) -> str:
    """Преобразует историю диалога в читаемый текстовый формат"""
    return "\n".join(f"[{entry['role']}] {entry['text']}" for entry in history)


class ReflectionAgent:
    def __init__(self):
        # LLM для анализа диалога
        self.llm = ChatTogether(model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
        # Модель для эмбеддингов
        self.embedding_model = SentenceTransformer("BAAI/bge-m3")
        # Клиент Qdrant
        self.qdrant = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY", None)
        )
        # Файл с историей диалога
        self.history_file = os.getenv("DIALOGUE_HISTORY_FILE", "dialogue_history.json")
        self.dialogue_history = load_dialogue_history(self.history_file)

        # Шаблон промпта для семантической памяти
        self.reflection_prompt = PromptTemplate(
            template="""
                    Ты — reflection-агент для call-центра. Твоя задача — извлечь из истории диалога полезные советы, действия и шаблоны поведения, которые можно переиспользовать позже.
                    
                    История диалога:
                    {history}
                    
                    Верни JSON с ключом "semantic_entries" — списком объектов с полями:
                    - "text": текст полезного совета или действия;
                    - "tags": массив ключевых слов или тем для поиска.
                    
                    Пример ответа:
                    {{
                      "semantic_entries": [
                        {{"text": "...", "tags": ["...", "..."]}},
                        ...
                      ]
                    }}
                    """,
            input_variables=["history"]
        )

    def save_history(self) -> None:
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.dialogue_history, f, ensure_ascii=False, indent=2)

    def reflect(self) -> List[Dict[str, Any]]:
        history_text = history_to_text(self.dialogue_history)
        chain = LLMChain(llm=self.llm, prompt=self.reflection_prompt)
        raw = chain.run(history=history_text)
        parsed = json.loads(raw)
        entries = parsed.get("semantic_entries", [])

        # Векторизуем и отправляем в Qdrant
        for entry in entries:
            vector = self.embedding_model.encode(entry["text"]).tolist()
            self.qdrant.upsert(
                collection_name="semantic_memory",
                points=[{
                    "id": uuid4().hex,
                    "vector": vector,
                    "payload": entry
                }]
            )
        return entries


if __name__ == "__main__":
    agent = ReflectionAgent()
    # Если нужно, предварительно загружаем историю (например, после звонка)
    entries = agent.reflect()
    print(json.dumps({"semantic_entries": entries}, ensure_ascii=False, indent=2))
