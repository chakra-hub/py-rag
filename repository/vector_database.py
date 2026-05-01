import chromadb
import uuid

class VectorRepository:
    def __init__(self,):
        self.client = chromadb.CloudClient(
            api_key='ck-6SnQiEtYypzMRqZMxtGY7xrcnnwk4YckkFiPn6uQyv1D',
            tenant='6a95ffa1-97bc-49a7-baf9-564d54e10f9d',
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