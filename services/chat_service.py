from repository.bm25_repository import BM25Repository
from repository.vector_database import VectorRepository
from services.ask_llm import AskLLM
from utils.createEmbeddings import createEmbeddings
from repository.semantic_cache import SemanticCacheRepository
import json
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
        vector_results = self.vector_db.query_text(question)[:2]
        bm25_results = self.bm25_db.query(question, n_results=2)
        seen = set()
        combined = []
        for chunk in vector_results + bm25_results:
            if chunk not in seen:
                seen.add(chunk)
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
            trace_name="rag-query",   
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

                context_text = "\n\n---\n\n".join(chunks)

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
            

    async def stream_response(self, session_id: str, question: str):
        query_embedding = createEmbeddings(question)
        cached = self.cache_repo.find_similar(query_embedding)

        if cached:
            answer = cached.get("response", "")
            for word in answer.split(" "):
                yield f"data: {json.dumps({'token': word + ' ', 'done': False, 'cached': True})}\n\n"
            yield f"data: {json.dumps({'token': '', 'done': True, 'session_id': session_id, 'cached': True})}\n\n"
            return

        chunks = self.hybrid_retrieve(question)

        if not chunks:
            yield f"data: {json.dumps({'token': 'I cannot find relevant information in the document.', 'done': True, 'session_id': session_id})}\n\n"
            return

        context_text = "\n\n---\n\n".join(chunks)
        full_response = ""

        async for chunk in self.ask_llm.stream_with_groq(
            session_id=session_id,
            question=question,
            context=context_text
        ):
            if '"done": false' in chunk or '"done":false' in chunk:
                data = json.loads(chunk.replace("data: ", "").strip())
                full_response += data.get("token", "")
            yield chunk

        self.cache_repo.save(
            query_embedding,
            question,
            {"response": full_response, "session_id": session_id}
        )