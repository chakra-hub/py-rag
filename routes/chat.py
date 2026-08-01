from fastapi import APIRouter
from pydantic import BaseModel
from services.chat_service import ChatService
from services.agentic_rag import run_agentic_rag
import uuid
from fastapi.responses import StreamingResponse


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

@router.post('/chat/agentic')
async def agentic_chat(request:ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    result = run_agentic_rag(
        session_id=session_id,
        question=request.question,
    )

    return {
        "response": result["answer"],
        "session_id": session_id,
        "final_query": result["query"],
        "attempts": result["rewrite_attempts"]
    }

@router.post('/chat/stream')
async def stream_chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    
    return StreamingResponse(
        chat_service.stream_response(
            session_id=session_id,
            question=request.question
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )