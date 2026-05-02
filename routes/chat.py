from fastapi import APIRouter
from pydantic import BaseModel
from services.chat_service import ChatService
import uuid

router = APIRouter()

class ChatRequest(BaseModel):
    session_id: str
    question: str
chat_service = ChatService()
@router.post('/chat')
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    return await chat_service.retrieve_and_ask_llm(
        session_id=session_id,
        question=request.question
    )