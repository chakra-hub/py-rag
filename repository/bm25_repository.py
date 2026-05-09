from rank_bm25 import BM25Okapi
import json, os

class BM25Repository:
    _instance = None

    def __new__(cls, storage_path=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, storage_path: str = None):
        if self._initialized:
            return  # Already initialized, skip
        
        if storage_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            storage_path = os.path.join(base_dir, "..", "bm25_store.json")
        self.storage_path = os.path.abspath(storage_path)
        self.chunks = []
        self.bm25 = None
        self._load()
        self._initialized = True

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
        self._save()

    def query(self, query: str, n_results: int = 3) -> list[str]:
        if not self.bm25 or not self.chunks:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
                
        mean_score = sum(scores) / len(scores)
        
        scored_chunks = list(zip(self.chunks, scores))
        filtered = [
            (chunk, score) for chunk, score in scored_chunks
            if score > mean_score and score > 0.5
        ]
                
        filtered.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, score in filtered[:n_results]]
    
    def query_raw(self, query: str, n_results: int = 3) -> list[str]:
        """No score filtering — returns top results regardless of score.
        Used by agentic RAG where grade_relevance node handles filtering."""
        if not self.bm25 or not self.chunks:
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        scored_chunks = list(zip(self.chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        return [chunk for chunk, score in scored_chunks[:n_results]]