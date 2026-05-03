from groq import Groq
from langchain_groq import ChatGroq
from pydantic import BaseModel
from config import settings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages
from core.langfuse_client import langfuse_handler

store = {}


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


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
                (
                    "system",
                    "You are a document assistant. Answer questions ONLY using the context provided. "
                    "If the answer is not found in the context, say 'I cannot find this information'. "
                    "Cite which part of the context supports your answer.",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "Context: {context}\n\nQuestion: {question}"),
            ]
        )
        self.trimmer = trim_messages(
            max_tokens=4096,             
            strategy="last",             
            token_counter=self.model,    
            include_system=True,         
            start_on="human",            
        )

        self.chain = self.prompt | self.trimmer | self.model

        self.with_history = RunnableWithMessageHistory(
            self.chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )

    async def chat_with_groq(self, session_id: str, question: str, context: str = ""):
        try:
            response = await self.with_history.ainvoke(
                {"question": question, "context": context},
                config={"configurable": {"session_id": session_id}, "callbacks": [langfuse_handler]},
            )

            return {"response": response.content, "session_id": session_id}

        except Exception as e:
            raise Exception(f"LLM error: {str(e)}")
