from fastapi import APIRouter
from pydantic import BaseModel
from app.rag.pipeline import rag_answer

router = APIRouter(prefix="/chat")

class ChatRequest(BaseModel):
    question: str

@router.post("/")
def chat(req: ChatRequest):
    answer = rag_answer(req.question)
    return {"answer": answer}
