from uuid import uuid4

from langchain_core.documents import Document
from docling.chunking import HybridChunker

from repository.bm25_repository import BM25Repository
from repository.vector_database import VectorRepository
from utils.createChunks import createChunks
from utils.extractDocsFromRequest import extractDocsFromRequest
from utils.buildMetadata import _build_metadata

chunker = HybridChunker()


class IngestService:

    def __init__(self):
        self.vector_db = VectorRepository()
        self.bm25_db = BM25Repository()

    def ingest_document(
        self,
        uploaded_file,
    ):

        if uploaded_file is None:
            raise ValueError("Document can't be empty")

        # Parse document using Docling
        result = extractDocsFromRequest(uploaded_file)
        doc = result.document

        # Structure-aware chunking
        chunks = createChunks(doc)

        if not chunks:
            raise ValueError("No chunks generated from the document.")

        # One doc id for the entire document
        document_id = str(uuid4())

        # Determine document name
        if isinstance(uploaded_file, str):
            doc_name = uploaded_file.split("/")[-1]
        else:
            doc_name = getattr(uploaded_file, "filename", "unknown")

        documents = []

        for index, chunk in enumerate(chunks):
            metadata =_build_metadata(
            chunk=chunk,
            document_id=document_id,
            chunk_index=index,
            )

            documents.append(
                Document(
                    page_content=chunker.contextualize(chunk),
                    metadata=metadata,
                )
            )

        ids = [str(uuid4()) for _ in documents]

        try:

            # Store in Chroma
            self.vector_db.add_documents(
                documents=documents,
                ids=ids,
            )

            # Store in BM25
            # (We'll make this dynamic later as well.)
            self.bm25_db.add_chunks(
            documents=documents,
        )

            return {
                "status": "success",
                "documents": len(documents),
            }

        except Exception as e:
            raise Exception(f"Ingestion failed: {str(e)}")