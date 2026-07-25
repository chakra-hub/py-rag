from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
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

    def _get_collection(
        self,
        database_name: str,
        collection_name: str,
        version: str
    ) -> Chroma:
        """
        Returns a Chroma collection instance.

        Collection naming convention:
        hr + v1 -> hr_v1
        engineering + v3 -> engineering_v3
        """

        return Chroma(
            collection_name=f"{collection_name}_{version}",
            embedding_function=self.model,
            chroma_cloud_api_key=settings.chroma_api_key,
            tenant=settings.chroma_tenant_id,
            database=database_name
        )

    def add_documents(
    self,
    database_name: str,
    collection_name: str,
    version: str,
    documents,
    ids
):
        db = self._get_collection(
            database_name,
            collection_name,
            version
        )

        BATCH_SIZE = 250  # safely below Chroma Cloud's limit

        total = len(documents)

        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)

            print(
                f"Inserting batch {start // BATCH_SIZE + 1} "
                f"({start}-{end - 1}) of {total}"
            )

            db.add_documents(
                documents=documents[start:end],
                ids=ids[start:end]
            )

        print(f"Successfully inserted {total} chunks.")

    def query_text(
        self,
        database_name: str,
        collection_name: str,
        version: str,
        query_text: str,
        n_results: int = 5
    ):
        db = self._get_collection(
            database_name,
            collection_name,
            version
        )

        results = db.similarity_search_with_score(
            query_text,
            k=n_results
        )

        filtered = [
            (doc.page_content, score)
            for doc, score in results
            if score < 1.1
        ]

        filtered.sort(key=lambda x: x[1])

        return [doc for doc, score in filtered]

    def query_text_raw(
        self,
        database_name: str,
        collection_name: str,
        version: str,
        query_text: str,
        n_results: int = 5
    ) -> list[str]:
        """
        No score filtering.
        Used by Agentic RAG where the relevance grader decides.
        """

        db = self._get_collection(
            database_name,
            collection_name,
            version
        )

        results = db.similarity_search_with_score(
            query_text,
            k=n_results
        )

        results.sort(key=lambda x: x[1])

        return [
            doc.page_content
            for doc, score in results
        ]

    def delete_collection(
        self,
        database_name: str,
        collection_name: str,
        version: str
    ):
        """
        Deletes a version if you ever want to remove it.
        """

        client = Chroma(
            collection_name=f"{collection_name}_{version}",
            embedding_function=self.model,
            chroma_cloud_api_key=settings.chroma_api_key,
            tenant=settings.chroma_tenant_id,
            database=database_name
        )

        client.delete_collection()

    def collection_exists(
        self,
        database_name: str,
        collection_name: str,
        version: str
    ) -> bool:
        """
        Checks whether a version already exists.
        """

        try:
            db = self._get_collection(
                database_name,
                collection_name,
                version
            )

            db.get()

            return True

        except Exception:
            return False