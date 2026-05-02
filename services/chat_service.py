from repository.bm25_repository import BM25Repository
from repository.vector_database import VectorRepository
from services.ask_llm import AskLLM
from utils.createEmbeddings import createEmbeddings

class ChatService:
    def __init__(self):
        self.vector_db = VectorRepository()
        self.ask_llm = AskLLM()
        self.bm25_db = BM25Repository()

    def hybrid_retrieve(self, question:str):
        question_embedding=createEmbeddings(question)
            

        retrieved_results = self.vector_db.query_text(question_embedding)
        vector_chunks = retrieved_results['documents'][0]

        bm25_chunks = self.bm25_db.query(question, n_results=3)
        print(vector_chunks,'vector chunks')
        print(bm25_chunks,'bm25 chunks')
        seen = set()
        combined = []
        for chunk in vector_chunks + bm25_chunks:
            if chunk not in seen:
                seen.add(chunk)
                combined.append(chunk)
            return combined[:5]

    async def retrieve_and_ask_llm(self, session_id: str, question: str):
        try:
            # Query vector database for context
            
            chunks = self.hybrid_retrieve(question)
            print(chunks)
            context_text="\n\n---\n\n".join(chunks)
            # Call LLM with question and context
            return await self.ask_llm.chat_with_groq(
                session_id=session_id,
                question=question,
                context=context_text
            )
        except Exception as e:
            raise Exception(f'Error: {str(e)}')