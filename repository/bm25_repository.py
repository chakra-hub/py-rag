import json
import os
import pickle
import re

from rank_bm25 import BM25Okapi


class BM25Repository:

    def __init__(self):
        self.base_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "indexes",
            )
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str):
        """Tokenize text consistently for indexing and querying."""
        return re.findall(r"\w+", text.lower())

    def _get_directory(
        self,
        database_name,
        collection_name,
        version,
    ):
        directory = os.path.join(
            self.base_path,
            database_name,
            collection_name,
            version,
        )

        os.makedirs(directory, exist_ok=True)

        return directory

    def _chunks_path(
        self,
        database_name,
        collection_name,
        version,
    ):
        return os.path.join(
            self._get_directory(
                database_name,
                collection_name,
                version,
            ),
            "chunks.json",
        )

    def _bm25_path(
        self,
        database_name,
        collection_name,
        version,
    ):
        return os.path.join(
            self._get_directory(
                database_name,
                collection_name,
                version,
            ),
            "bm25.pkl",
        )

    # ------------------------------------------------------------------
    # Chunk Storage
    # ------------------------------------------------------------------

    def _load_chunks(
        self,
        database_name,
        collection_name,
        version,
    ):

        path = self._chunks_path(
            database_name,
            collection_name,
            version,
        )

        if not os.path.exists(path):
            return []

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_chunks(
        self,
        database_name,
        collection_name,
        version,
        chunks,
    ):

        path = self._chunks_path(
            database_name,
            collection_name,
            version,
        )

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                chunks,
                f,
                ensure_ascii=False,
                indent=2,
            )

    # ------------------------------------------------------------------
    # BM25 Storage
    # ------------------------------------------------------------------

    def _save_bm25(
        self,
        database_name,
        collection_name,
        version,
        bm25,
    ):

        with open(
            self._bm25_path(
                database_name,
                collection_name,
                version,
            ),
            "wb",
        ) as f:
            pickle.dump(bm25, f)

    def _load_bm25(
        self,
        database_name,
        collection_name,
        version,
    ):

        path = self._bm25_path(
            database_name,
            collection_name,
            version,
        )

        if not os.path.exists(path):
            return None

        with open(path, "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        database_name,
        collection_name,
        version,
        documents,
    ):

        chunks = self._load_chunks(
            database_name,
            collection_name,
            version,
        )

        chunks.extend(
            [
                doc.page_content
                for doc in documents
            ]
        )

        self._save_chunks(
            database_name,
            collection_name,
            version,
            chunks,
        )

        tokenized_chunks = [
            self._tokenize(chunk)
            for chunk in chunks
        ]

        bm25 = BM25Okapi(tokenized_chunks)

        self._save_bm25(
            database_name,
            collection_name,
            version,
            bm25,
        )

    def query(
        self,
        database_name,
        collection_name,
        version,
        query,
        n_results=3,
    ):

        chunks = self._load_chunks(
            database_name,
            collection_name,
            version,
        )

        if not chunks:
            return []

        bm25 = self._load_bm25(
            database_name,
            collection_name,
            version,
        )

        if bm25 is None:
            return []

        scores = bm25.get_scores(
            self._tokenize(query)
        )

        mean_score = sum(scores) / len(scores)

        scored = [
            (chunk, score)
            for chunk, score in zip(chunks, scores)
            if score > mean_score and score > 0.5
        ]

        scored.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            chunk
            for chunk, score in scored[:n_results]
        ]

    def query_raw(
        self,
        database_name,
        collection_name,
        version,
        query,
        n_results=3,
    ):

        chunks = self._load_chunks(
            database_name,
            collection_name,
            version,
        )

        if not chunks:
            return []

        bm25 = self._load_bm25(
            database_name,
            collection_name,
            version,
        )

        if bm25 is None:
            return []

        scores = bm25.get_scores(
            self._tokenize(query)
        )

        scored = list(zip(chunks, scores))

        scored.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            chunk
            for chunk, score in scored[:n_results]
        ]