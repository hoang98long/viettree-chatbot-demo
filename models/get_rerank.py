# scripts/download_rerank.py
from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LOCAL_PATH = "models/rerank/ms-marco-minilm"

CrossEncoder(MODEL_NAME, cache_folder=LOCAL_PATH)

print("✅ Re-rank model downloaded")
