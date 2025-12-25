# scripts/download_embedding.py
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-base"
LOCAL_PATH = "models/embeddings/multilingual-e5-base"

model = SentenceTransformer(
    MODEL_NAME,
    cache_folder=LOCAL_PATH
)

print("✅ Embedding model downloaded to:", LOCAL_PATH)
