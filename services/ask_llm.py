from groq import RateLimitError
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from pydantic import BaseModel

from config import settings


store = {}
MAX_HISTORY = 5


class LLMResponse(BaseModel):
    success: bool
    content: str | None = None
    error: str | None = None


class ChatRequest(BaseModel):
    session_id: str
    question: str
    context: str | None = None


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    history = store[session_id]

    history.messages = history.messages[-MAX_HISTORY:]

    return history


class AskLLM:
    def __init__(self):
        self.model = ChatGroq(
            groq_api_key=settings.groq_api_key,
            model_name="llama-3.3-70b-versatile",
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a document assistant.\n"
                    "Answer ONLY using the provided context.\n"
                    "If the context is irrelevant or does not contain the answer, say:\n"
                    "'I cannot find this information in the provided documents.'\n"
                    "DO NOT use prior knowledge.",
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

    # ------------------------------------------------------------------
    # Generic synchronous invoke
    # ------------------------------------------------------------------

    def invoke(self, prompt: str) -> LLMResponse:
        try:
            response = self.model.invoke(prompt)

            return LLMResponse(
                success=True,
                content=response.content,
            )

        except RateLimitError:
            return LLMResponse(
                success=False,
                error=(
                    "Groq API rate limit exceeded. "
                    "Please try again after some time."
                ),
            )

        except Exception as e:
            return LLMResponse(
                success=False,
                error=f"LLM Error: {str(e)}",
            )

    # ------------------------------------------------------------------
    # Chat with history
    # ------------------------------------------------------------------

    async def chat_with_groq(
        self,
        session_id: str,
        question: str,
        context: str,
        callbacks,
    ):
        try:
            response = await self.with_history.ainvoke(
                {
                    "question": question,
                    "context": context,
                },
                config={
                    "configurable": {
                        "session_id": session_id
                    },
                    "callbacks": callbacks,
                },
            )

            return {
                "response": response.content,
                "session_id": session_id,
            }

        except RateLimitError:
            return {
                "response": (
                    "Groq API rate limit exceeded. "
                    "Please try again after some time."
                ),
                "session_id": session_id,
            }

        except Exception as e:
            raise Exception(f"LLM error: {str(e)}")

    # ------------------------------------------------------------------
    # Streaming chat
    # ------------------------------------------------------------------

    async def stream_with_groq(
        self,
        session_id: str,
        question: str,
        context: str,
    ):
        import json

        try:
            async for chunk in self.with_history.astream(
                {
                    "question": question,
                    "context": context,
                },
                config={
                    "configurable": {
                        "session_id": session_id
                    }
                },
            ):
                token = chunk.content

                if token:
                    yield (
                        f"data: {json.dumps({'token': token, 'done': False})}\n\n"
                    )

            yield (
                f"data: {json.dumps({'token': '', 'done': True, 'session_id': session_id})}\n\n"
            )

        except RateLimitError:
            yield (
                f"data: {json.dumps({'token': 'Groq API rate limit exceeded. Please try again after some time.', 'done': True})}\n\n"
            )

        except Exception as e:
            yield (
                f"data: {json.dumps({'token': f'LLM Error: {str(e)}', 'done': True})}\n\n"
            )