from fastapi import APIRouter
from pydantic import BaseModel
from services.chat_service import ChatService

router = APIRouter()

class ChatRequest(BaseModel):
    session_id: str=None
    question: str
chat_service = ChatService()
@router.post('/chat')
async def chat(request: ChatRequest):

    return await chat_service.retrieve_and_ask_llm(
        session_id=request.session_id,
        question=request.question
    )