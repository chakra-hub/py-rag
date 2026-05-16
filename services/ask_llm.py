from langchain_groq import ChatGroq
from pydantic import BaseModel
from config import settings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}


MAX_HISTORY = 5

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    history = store[session_id]

    history.messages = history.messages[-MAX_HISTORY:]

    return history


class ChatRequest(BaseModel):
    session_id: str
    question: str
    context: str = None


class AskLLM:
    def __init__(self):
        self.model = ChatGroq(
            groq_api_key=settings.groq_api_key, model_name="llama-3.3-70b-versatile"
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
 "You are a document assistant.\n"
 "Answer ONLY using the provided context.\n"
 "If the context is irrelevant or does not contain the answer, say:\n"
 "'I cannot find this information in the provided documents.'\n"
 "DO NOT use prior knowledge."
),
                MessagesPlaceholder(variable_name="history"),
                ("human", "Context: {context}\n\nQuestion: {question}"),
            ]
        )

        self.chain = self.prompt | self.model

        self.with_history = RunnableWithMessageHistory(
            self.chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )

    async def chat_with_groq(self, session_id: str, question: str, context: str, callbacks):
        try:
            response = await self.with_history.ainvoke(
                {"question": question, "context": context},
                config={"configurable": {"session_id": session_id},'callbacks':callbacks},
            )

            return {"response": response.content, "session_id": session_id}

        except Exception as e:
            raise Exception(f"LLM error: {str(e)}")
        
    async def stream_with_groq(self, session_id: str, question: str, context: str):
        import json
        
        full_response = ""
        
        async for chunk in self.with_history.astream(
            {"question": question, "context": context},
            config={"configurable": {"session_id": session_id}}
        ):
            token = chunk.content
            if token:
                full_response += token
                yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"
        
        yield f"data: {json.dumps({'token': '', 'done': True, 'session_id': session_id})}\n\n"
