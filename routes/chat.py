from fastapi import APIRouter
from pydantic import BaseModel
from services.chat_service import ChatService
from services.agentic_rag import rag_graph
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

@router.post('/chat/agentic')
async def agentic_chat(request:ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    result = rag_graph.invoke(
    {
        "question":request.question,
        "query":request.question,
        "answer":"",
        "chunks":[],
        "is_relevant":False,
        "rewrite_attempts":0,
        "session_id":request.session_id
    })

    return {
        "response": result["answer"],
        "session_id": session_id,
        "final_query": result["query"],
        "attempts": result["rewrite_attempts"]
    }