from fastapi import APIRouter
from pydantic import BaseModel
from app.rag.agentic import agentic_answer

router = APIRouter(prefix="/chat")

class ChatRequest(BaseModel):
    question: str


@router.post("/")
def chat(req: ChatRequest):
    answer = agentic_answer(req.question)
    return {"answer": answer}
