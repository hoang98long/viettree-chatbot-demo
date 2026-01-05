from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

LLM_MODEL_PATH = BASE_DIR / "models/llm"
EMBEDDING_MODEL_PATH = BASE_DIR / "models/embedding"

DOCS_PATH = BASE_DIR / "data/docs"
FAISS_PATH = BASE_DIR / "data/faiss"

TOP_K = 3
MAX_NEW_TOKENS = 300

CHUNK_PRESETS = {
    "faq": {
        "chunk_size": 300,
        "chunk_overlap": 50
    },
    "sop": {
        "chunk_size": 600,
        "chunk_overlap": 100
    },
    "manual": {
        "chunk_size": 800,
        "chunk_overlap": 150
    },
    "report": {
        "chunk_size": 1000,
        "chunk_overlap": 200
    }
}
