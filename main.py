from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

qdrant = QdrantClient(url="http://localhost:6333")

# Создаём коллекцию (назовём "kb")
qdrant.recreate_collection(
    collection_name="kb",
    vectors_config=VectorParams(size=embeddings[0].shape[0], distance=Distance.COSINE)
)

# Подготавливаем точки
points = [
    PointStruct(id=d["id"], vector=emb, payload={"text": d["text"], "source_query": d["source_query"]})
    for d, emb in zip(docs, embeddings)
]

# Загружаем
qdrant.upsert(collection_name="kb", points=points)
