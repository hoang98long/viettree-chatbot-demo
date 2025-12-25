from app.config import DOCS_PATH, FAISS_PATH, EMBEDDING_MODEL_PATH
from app.rag.embedding import load_embedding_model
from app.rag.vector_store import VectorStore

def load_docs():
    texts = []
    for file in DOCS_PATH.glob("*.txt"):
        texts.append(file.read_text(encoding="utf-8"))
    return texts

if __name__ == "__main__":
    docs = load_docs()
    model = load_embedding_model(str(EMBEDDING_MODEL_PATH))

    vectors = model.encode(docs)

    store = VectorStore(dim=vectors.shape[1])
    store.add(vectors, docs)
    store.save(str(FAISS_PATH))

    print("✅ Ingest xong tài liệu")
