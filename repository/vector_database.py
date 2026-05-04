import chromadb
import uuid

from langchain_huggingface import HuggingFaceEmbeddings
from config import settings
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer

class VectorRepository:
    def __init__(self,):
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
        print(results,'result direct')
        filtered_documents = [
            doc.page_content
            for doc, score in results
            if score < 1.0  # Lower distance = more similar (was 0.5, too strict)
        ]

        return filtered_documents