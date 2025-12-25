from app.config import (
    EMBEDDING_MODEL_PATH,
    FAISS_PATH,
    TOP_K,
    MAX_NEW_TOKENS
)
from app.rag.embedding import load_embedding_model
from app.rag.vector_store import VectorStore
from app.llm.llm_loader import load_llm
import torch

# Load models ONCE
embedding_model = load_embedding_model(str(EMBEDDING_MODEL_PATH))
vector_store = VectorStore(dim=768)
vector_store.load(str(FAISS_PATH))

tokenizer, model = load_llm(str("models/llm"))

SYSTEM_PROMPT = """Bạn là trợ lý AI.
Chỉ trả lời dựa trên nội dung tài liệu được cung cấp.
Nếu không có thông tin, hãy nói "Tôi không tìm thấy thông tin trong tài liệu".
"""

def rag_answer(question: str) -> str:
    q_vec = embedding_model.encode([question])
    docs = vector_store.search(q_vec, TOP_K)

    context = "\n\n".join(docs)

    prompt = f"""
    {SYSTEM_PROMPT}
    
    TÀI LIỆU:
    {context}
    
    CÂU HỎI:
    {question}
    
    TRẢ LỜI:
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.3
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
