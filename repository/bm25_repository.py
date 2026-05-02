from rank_bm25 import BM25Okapi
import json, os

class BM25Repository:
    def __init__(self, storage_path: str = "bm25_store.json"):
        self.storage_path = storage_path
        self.chunks = []
        self.bm25 = None
        self._load()  # Load from disk on startup

    def _load(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r") as f:
                self.chunks = json.load(f)
            self._rebuild_index()

    def _save(self):
        with open(self.storage_path, "w") as f:
            json.dump(self.chunks, f)

    def _rebuild_index(self):
        if self.chunks:
            tokenized = [chunk.lower().split() for chunk in self.chunks]
            self.bm25 = BM25Okapi(tokenized)

    def add_chunks(self, chunks: list[str]):
        self.chunks.extend(chunks)
        self._rebuild_index()
        self._save()  # Persist to disk — fixes the memory problem

    def query(self, query: str, n_results: int = 3) -> list[str]:
        if not self.bm25 or not self.chunks:
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)),
                           key=lambda i: scores[i],
                           reverse=True)[:n_results]
        return [self.chunks[i] for i in top_indices]