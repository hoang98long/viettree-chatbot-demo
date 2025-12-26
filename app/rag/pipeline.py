from app.config import (
    EMBEDDING_MODEL_PATH,
    FAISS_PATH,
    TOP_K,
    MAX_NEW_TOKENS,
    LLM_MODEL_PATH
)
from app.rag.embedding import load_embedding_model
from app.rag.vector_store import VectorStore
from app.llm.llm_loader import load_llm
import torch
import os

_embedding_model = None
_vector_store = None
_tokenizer = None
_model = None


def init_rag():
    global _embedding_model, _vector_store, _tokenizer, _model

    if _embedding_model is None:
        _embedding_model = load_embedding_model(str(EMBEDDING_MODEL_PATH))

    if _vector_store is None:
        _vector_store = VectorStore(dim=768)
        if not os.path.exists(FAISS_PATH / "index.faiss"):
            raise RuntimeError(
                "❌ FAISS index chưa tồn tại. Hãy chạy: python -m scripts.ingest"
            )
        _vector_store.load(str(FAISS_PATH))

    if _tokenizer is None or _model is None:
        _tokenizer, _model = load_llm(str(LLM_MODEL_PATH))


def rag_answer(question: str) -> str:
    init_rag()

    q_vec = _embedding_model.encode([question])
    docs = _vector_store.search(q_vec, TOP_K)

    context = "\n\n".join(docs)

    prompt = f"""
Bạn là trợ lý AI.
Chỉ trả lời dựa trên nội dung tài liệu.
Nếu không có, nói: "Không tìm thấy thông tin trong tài liệu".

TÀI LIỆU:
{context}

CÂU HỎI:
{question}

TRẢ LỜI:
"""

    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.3
        )

    return _tokenizer.decode(outputs[0], skip_special_tokens=True)
