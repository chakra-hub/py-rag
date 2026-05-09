import chromadb
import uuid

from langchain_huggingface import HuggingFaceEmbeddings
from config import settings
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer

class VectorRepository:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        # your existing __init__ code here
        self._initialized = True
        self.model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.client = Chroma(
            collection_name='document-source',
            embedding_function=self.model,
            chroma_cloud_api_key=settings.chroma_api_key,
            tenant=settings.chroma_tenant_id,
            database='py-rag'
        )

    def add_document(self, documents, ids):
        self.client.add_documents(
            documents=documents,
            ids=ids
        )

    def query_text(self, query_text: str, n_results: int = 5):
        results = self.client.similarity_search_with_score(query_text, k=n_results)
        
        filtered = [
            (doc.page_content, score)
            for doc, score in results
            if score < 1.1
        ]
        
        # Sort ascending — lowest score = most similar = best result first
        filtered.sort(key=lambda x: x[1])
        return [doc for doc, score in filtered]
    
    def query_text_raw(self, query_text: str, n_results: int = 5) -> list[str]:
        """No score filtering — returns top results regardless of score.
        Used by agentic RAG where grade_relevance node handles filtering."""
        results = self.client.similarity_search_with_score(query_text, k=n_results)
        results.sort(key=lambda x: x[1]) 
        return [doc.page_content for doc, score in results]
