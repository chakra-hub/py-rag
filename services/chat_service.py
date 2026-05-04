from repository.bm25_repository import BM25Repository
from repository.vector_database import VectorRepository
from services.ask_llm import AskLLM
from utils.createEmbeddings import createEmbeddings
from repository.semantic_cache import SemanticCacheRepository

from langfuse import observe, propagate_attributes
from langfuse.langchain import CallbackHandler


class ChatService:
    def __init__(self):
        self.vector_db = VectorRepository()
        self.ask_llm = AskLLM()
        self.bm25_db = BM25Repository()
        self.cache_repo = SemanticCacheRepository()

    @observe(name="hybrid-retrieval")
    def hybrid_retrieve(self, question: str) -> list[str]:

        vector_results = self.vector_db.query_text(question)
        print(vector_results,'vector results')
        bm25_results = self.bm25_db.query(question, n_results=3)

        combined = vector_results.copy()

        for chunk in bm25_results:
            if chunk not in combined:
                combined.append(chunk)

        return combined[:3]

    @observe(name="cache-lookup")
    def check_cache(self, embedding):
        result = self.cache_repo.find_similar(embedding)
        return {
            "hit": result is not None,
            "answer": result
        }

    @observe(name="rag-pipeline")
    async def retrieve_and_ask_llm(self, session_id: str, question: str):

        with propagate_attributes(
            trace_name="rag-query",   # 👈 important for naming
            session_id=session_id,
            user_id="user-123",
        ):
            try:
                query_embedding = createEmbeddings(question)

                cache_result = self.check_cache(query_embedding)
                
                if cache_result["hit"]:
                    return cache_result["answer"]

                chunks = self.hybrid_retrieve(question)

                if not chunks:
                    return {
                        "response": "I cannot find this information in the provided documents.",
                        "session_id": session_id
                    }

                context_text = "\n\n---\n\n".join([c[:300] for c in chunks])

                handler = CallbackHandler()

                answer = await self.ask_llm.chat_with_groq(
                    session_id=session_id,
                    question=question,
                    context=context_text,
                    callbacks=[handler]
                )

                self.cache_repo.save(query_embedding, question, answer)

                return answer

            except Exception as e:
                raise Exception(f"Error: {str(e)}")