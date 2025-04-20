import os
import json
from collections import Counter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 2) Конфигурация Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
collection_name = "support_intents"

# Подключаемся к Qdrant
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# 3) Инициализация модели эмбеддингов из Hugging Face
# Рекомендуемая модель для качества семантического поиска: all-mpnet-base-v2
model_name = "BAAI/bge-m3"
model = SentenceTransformer(model_name)
vector_size = model.get_sentence_embedding_dimension()

# Проверка на дублирование ID в final.json
def check_duplicate_ids(records):
    counts = Counter(rec["id"] for rec in records)
    duplicates = [i for i, c in counts.items() if c > 1]
    if duplicates:
        print(f"Предупреждение: Найдены дублирующиеся id: {duplicates[:10]}")
    return counts

# Создание или перезапись коллекции
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
)

# 4) Загрузка данных из final.json
with open("final.json", encoding="utf-8") as f:
    records = json.load(f)

# Проверяем на дублированные ID
check_duplicate_ids(records)

a = 0

batch_size = 64
points = []
for rec in tqdm(records, desc="Embedding intents"):
    name = rec["name"]
    content = rec.get("content", "")
    a += 1

    # Объединяем name и content для генерации эмбеддинга
    combined_text = f"{name} {content}".strip()  # Удаляем лишние пробелы, если content пуст

    # Генерация эмбеддинга на объединенном тексте
    vector = model.encode(combined_text, convert_to_numpy=True).tolist()

    # Подготовка payload (без изменений)
    payload = {
        "name": name,
        "content": content,
        "productLogoLink": rec.get("productLogoLink"),
        "urlArticleOnSupport": rec.get("urlArticleOnSupport"),
    }
    points.append({"id": a, "vector": vector, "payload": payload})

# 6) Логирование и загрузка векторов партиями с ожиданием подтверждения
for i in range(0, len(points), batch_size):
    batch = points[i: i + batch_size]
    try:
        # Загружаем батч в Qdrant с ожиданием подтверждения
        resp = client.upsert(collection_name=collection_name, points=batch, wait=True)
        print(f"Batch {i//batch_size + 1}: успешно загружено {len(batch)} точек.")
    except Exception as e:
        print(f"Batch {i//batch_size + 1}: ошибка при загрузке —", e)

print("✅ Data ingested into Qdrant using Hugging Face model successfully.")

# 7) Пример функции семантического поиска
def semantic_search(query: str, top_k: int = 5) -> list:
    """
    Поиск наиболее близких intent'ов по векторному сходству.
    Возвращает топ-k результатов с payload и score.
    """
    q_vector = model.encode(query, convert_to_numpy=True).tolist()
    hits = client.search(
        collection_name=collection_name,
        query_vector=q_vector,
        limit=top_k,
    )
    results = []
    for hit in hits:
        results.append({
            "id": hit.id,
            "score": hit.score,
            **hit.payload,
        })
    return results

