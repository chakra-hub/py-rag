from utils.createChunks import createChunks
from utils.createEmbeddings import createEmbeddings
from utils.extractDocsFromRequest import extractDocsFromRequest
from repository.vector_database import VectorRepository

class IngestService:
    def __init__(self):
        self.vector_db = VectorRepository()

    def ingest_document(self, docs):
        if not docs:
            raise Exception("Document can't be empty")

        extracted_docs = extractDocsFromRequest(docs)
        doc_text=extracted_docs.document.export_to_markdown()
        chunks=createChunks(doc_text)
        embeddings=createEmbeddings(chunks)
        try:
            self.vector_db.add_document(embeddings, chunks)
        except Exception as e:
            raise Exception(f'Ingestion failed:{str(e)}')
    