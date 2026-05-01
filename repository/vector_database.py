import chromadb
import uuid
from config import settings

class VectorRepository:
    def __init__(self,):
        self.client = chromadb.CloudClient(
            api_key=settings.chroma_api_key,
            tenant=settings.chroma_tenant_id,
            database='py-rag'
            )
        self.collection=self.client.get_or_create_collection(
            name='document-source',
        )

    def add_document(self, embeddings:list[int], chunks:list[str]):
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[str(uuid.uuid4()) for _ in chunks],
    )


    def query_text(self, query_embeddings):
        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=3
        )