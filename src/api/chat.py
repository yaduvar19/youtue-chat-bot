from fastapi import APIRouter
from pydantic import BaseModel
from src.services.rag_service import generate_answer

router = APIRouter()

class ChatRequest(BaseModel):
    question: str
    video_id: str

@router.post("/ask")
def ask_question(data: ChatRequest):
    answer = generate_answer(data.question, data.video_id)
    return {"answer": answer}