from repository.bm25_repository import BM25Repository
from repository.vector_database import VectorRepository
from services.ask_llm import AskLLM
from utils.createEmbeddings import createEmbeddings
from repository.semantic_cache import SemanticCacheRepository
from core.langfuse_client import langfuse_client

class ChatService:
    def __init__(self):
        self.vector_db = VectorRepository()
        self.ask_llm = AskLLM()
        self.bm25_db = BM25Repository()
        self.cache_repo = SemanticCacheRepository()

    def hybrid_retrieve(self, question: str, question_embedding: list) -> list[str]:
        retrieved_results = self.vector_db.query_text(question)
        print(retrieved_results,'Results')
        bm25_chunks = self.bm25_db.query(question, n_results=3)
        
        seen = set()
        combined = []
        for chunk in retrieved_results + bm25_chunks:
            if chunk not in seen:
                seen.add(chunk)
                combined.append(chunk)
        return combined[:5] 

    async def retrieve_and_ask_llm(self, session_id: str, question: str):
        try:
            query_embedding = createEmbeddings(question)
            
            cached = self.cache_repo.find_similar(query_embedding)
            print(cached,'**********CACHED**************')
            if cached:
                return cached
            
            chunks = self.hybrid_retrieve(question, query_embedding)
            context_text = "\n\n---\n\n".join(chunks)
            print('**********LLM CALL**************')
            
            answer = await self.ask_llm.chat_with_groq(
                session_id=session_id,
                question=question,
                context=context_text
            )
            self.cache_repo.save(query_embedding, question, answer)
            return answer
        except Exception as e:
            raise Exception(f'Error: {str(e)}')