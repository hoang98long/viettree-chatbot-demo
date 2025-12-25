from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

LLM_MODEL_PATH = BASE_DIR / "models/llm"
EMBEDDING_MODEL_PATH = BASE_DIR / "models/embedding"

DOCS_PATH = BASE_DIR / "data/docs"
FAISS_PATH = BASE_DIR / "data/faiss"

TOP_K = 3
MAX_NEW_TOKENS = 300
