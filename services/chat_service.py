from repository.vector_database import VectorRepository
from services.ask_llm import AskLLM
from utils.createEmbeddings import createEmbeddings

class ChatService:
    def __init__(self):
        self.vector_db = VectorRepository()
        self.ask_llm = AskLLM()

    async def retrieve_and_ask_llm(self, session_id: str, question: str):
        try:
            # Query vector database for context
            question_embedding=createEmbeddings(question)
            

            retrieved_results = self.vector_db.query_text(question_embedding)
            # Call LLM with question and context
            return await self.ask_llm.chat_with_groq(
                session_id=session_id,
                question=question,
                context=retrieved_results['documents'][0]
            )
        except Exception as e:
            raise Exception(f'Error: {str(e)}')