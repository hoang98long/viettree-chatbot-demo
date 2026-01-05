from app.config import (
    DOCS_PATH,
    FAISS_PATH,
    EMBEDDING_MODEL_PATH,
    CHUNK_PRESETS
)
from app.rag.embedding import load_embedding_model
from app.rag.vector_store import VectorStore
from app.rag.chunking import simple_chunk_text
import argparse


def load_and_chunk_docs(doc_type: str):
    if doc_type not in CHUNK_PRESETS:
        raise ValueError(f"❌ Unknown doc_type: {doc_type}")

    preset = CHUNK_PRESETS[doc_type]
    chunk_size = preset["chunk_size"]
    chunk_overlap = preset["chunk_overlap"]

    all_chunks = []

    for file in DOCS_PATH.glob("*.txt"):
        content = file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        chunks = simple_chunk_text(
            content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        all_chunks.extend(chunks)

    print(f"📄 Tổng số chunk tạo ra: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--doc_type",
        type=str,
        default="sop",
        help="Loại tài liệu: faq | sop | manual | report"
    )
    args = parser.parse_args()

    docs = load_and_chunk_docs(args.doc_type)

    if not docs:
        raise RuntimeError("❌ Không tạo được chunk nào")

    embedding_model = load_embedding_model(str(EMBEDDING_MODEL_PATH))
    vectors = embedding_model.encode(docs)

    store = VectorStore(dim=vectors.shape[1])
    store.add(vectors, docs)
    store.save(str(FAISS_PATH))

    print("✅ Ingest + chunking hoàn tất")
