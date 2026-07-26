from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings


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

        self._initialized = True

        self.model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        self.db = Chroma(
        collection_name=f"{settings.chroma_collection}_{settings.chroma_collection_version}",
        embedding_function=self.model,
        chroma_cloud_api_key=settings.chroma_api_key,
        tenant=settings.chroma_tenant_id,
        database=settings.chroma_database,
    )

    def _get_collection(self):
        return self.db

    def add_documents(self, documents, ids):
        db = self._get_collection()

        BATCH_SIZE = 250
        total = len(documents)

        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)

            print(
                f"Inserting batch {start // BATCH_SIZE + 1} "
                f"({start}-{end - 1}) of {total}"
            )

            db.add_documents(
                documents=documents[start:end],
                ids=ids[start:end],
            )

        print(f"Successfully inserted {total} chunks.")

    def query_text(
        self,
        query_text: str,
        n_results: int = 5,
    ):
        """
        Returns page contents only.
        Used by traditional RAG.
        """

        db = self._get_collection()

        results = db.similarity_search_with_score(
            query=query_text,
            k=n_results,
        )

        filtered = [
            (doc.page_content, score)
            for doc, score in results
            if score < 1.1
        ]

        filtered.sort(key=lambda x: x[1])

        return [doc for doc, _ in filtered]

    def query_text_raw(
        self,
        query_text: str,
        n_results: int = 50,
    ):
        """
        Returns LangChain Document objects.
        Used by Agentic RAG.
        No score filtering.
        """

        db = self._get_collection()

        results = db.similarity_search_with_score(
            query=query_text,
            k=n_results,
        )

        return [doc for doc, _ in results]

    def delete_collection(self):
        self._get_collection().delete_collection()

    def collection_exists(self) -> bool:
        try:
            self._get_collection().get()
            return True
        except Exception:
            return False