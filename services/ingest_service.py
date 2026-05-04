from utils.createChunks import createChunks
from utils.extractDocsFromRequest import extractDocsFromRequest
from repository.vector_database import VectorRepository
from repository.bm25_repository import BM25Repository
from langchain_core.documents import Document
from uuid import uuid4

class IngestService:
    def __init__(self):
        self.vector_db = VectorRepository()
        self.bm25_db = BM25Repository()  # Add this

    def ingest_document(self, docs):
        if not docs:
            raise Exception("Document can't be empty")

        extracted_docs = extractDocsFromRequest(docs)
        doc_text = extracted_docs.document.export_to_markdown()
        chunks = createChunks(doc_text)
        
        documents = [
            Document(
                page_content=chunk,
                metadata={"source": "uploaded_document"},
                id=str(uuid4())
            )
            for chunk in chunks
        ]
        
        uuids = [str(uuid4()) for _ in documents]
        
        try:
            self.vector_db.add_document(documents, uuids)
            self.bm25_db.add_chunks(chunks)  # Add this
        except Exception as e:
            raise Exception(f'Ingestion failed:{str(e)}')