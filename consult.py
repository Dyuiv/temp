import os
import argparse
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from docx import Document
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize

# Если ещё не скачано, раскомментируйте:
nltk.download('punkt')
nltk.download('punkt_tab')

# Настройки (можно переопределить через окружение)
QDRANT_URL            = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY        = os.getenv("QDRANT_API_KEY", None)
EMBEDDING_MODEL_NAME  = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
COLLECTION_NAME       = os.getenv("QA_COLLECTION_NAME", "qa_rules")
BATCH_SIZE            = int(os.getenv("BATCH_SIZE", 64))

# Инициализация
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=False
)

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
vector_size = embedding_model.get_sentence_embedding_dimension()

def create_collection(client: QdrantClient, name: str):
    """
    (Re)create a Qdrant collection with cosine distance.
    """
    client.recreate_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    print(f"[+] Collection '{name}' created (vector size={vector_size}).")

def index_docx_rules(docx_path: str, client: QdrantClient, collection: str):
    """
    Read .docx file, split into sentences, embed them and upsert into Qdrant.
    """
    # 1) Load document and split into sentences
    doc = Document(docx_path)
    sentences = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        for sent in sent_tokenize(text):
            sent = sent.strip()
            if sent:
                sentences.append(sent)
    print(f"[+] Extracted {len(sentences)} sentences from '{docx_path}'.")

    # 2) Prepare PointStruct objects
    points = []
    for idx, sentence in enumerate(sentences, start=1):
        vector = embedding_model.encode(sentence, convert_to_numpy=True).tolist()
        payload = {"text": sentence}
        points.append(PointStruct(id=idx, vector=vector, payload=payload))

    # 3) Upsert in batches
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i : i + BATCH_SIZE]
        client.upsert(collection_name=collection, points=batch, wait=True)
        print(f"    • Upserted batch {i//BATCH_SIZE + 1}: {len(batch)} items.")
    print("[+] Indexing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index DOCX rules into Qdrant")
    parser.add_argument("docx_path", help="Path to the .docx file with rules")
    args = parser.parse_args()

    # 1) create or reset collection
    create_collection(qdrant_client, COLLECTION_NAME)
    # 2) index sentences
    index_docx_rules(args.docx_path, qdrant_client, COLLECTION_NAME)
