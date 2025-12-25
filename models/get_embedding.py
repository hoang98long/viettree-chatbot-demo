# scripts/download_embedding.py
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"
LOCAL_PATH = "models/embeddings/bge-m3"

model = SentenceTransformer(
    MODEL_NAME,
    cache_folder=LOCAL_PATH
)

print("✅ Embedding model downloaded to:", LOCAL_PATH)
