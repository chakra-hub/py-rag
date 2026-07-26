import json
import os
import pickle
import re
from langchain_core.documents import Document
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
    ):
        directory = os.path.join(
            self.base_path,
        )

        os.makedirs(directory, exist_ok=True)

        return directory

    def _chunks_path(
        self,
    ):
        return os.path.join(
            self._get_directory(
            ),
            "chunks.json",
        )

    def _bm25_path(
        self,

    ):
        return os.path.join(
            self._get_directory(
            ),
            "bm25.pkl",
        )

    # ------------------------------------------------------------------
    # Chunk Storage
    # ------------------------------------------------------------------

    def _load_chunks(
        self,

    ):

        path = self._chunks_path(
        )

        if not os.path.exists(path):
            return []

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_chunks(
        self,
        chunks,
    ):

        path = self._chunks_path(
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
        bm25,
    ):

        with open(
            self._bm25_path(
            ),
            "wb",
        ) as f:
            pickle.dump(bm25, f)

    def _load_bm25(
        self,

    ):

        path = self._bm25_path(

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

        documents,
    ):

        chunks = self._load_chunks(

        )

        chunks.extend(
    [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in documents
    ]
)

        self._save_chunks(

            chunks,
        )

        tokenized_chunks = [
    self._tokenize(chunk["page_content"])
    for chunk in chunks
]

        bm25 = BM25Okapi(tokenized_chunks)

        self._save_bm25(

            bm25,
        )

    def query(
        self,

        query,
        n_results=3,
    ):

        chunks = self._load_chunks(

        )

        if not chunks:
            return []

        bm25 = self._load_bm25(

        )

        if bm25 is None:
            return []

        scores = bm25.get_scores(
            self._tokenize(query)
        )

        mean_score = sum(scores) / len(scores)

        scored = [
            (chunk["page_content"], score)
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

        query,
        n_results=3,
    ):

        chunks = self._load_chunks(

        )

        if not chunks:
            return []

        bm25 = self._load_bm25(

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
    Document(
        page_content=chunk["page_content"],
        metadata=chunk["metadata"],
    )
    for chunk, score in scored[:n_results]
]