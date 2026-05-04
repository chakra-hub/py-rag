import numpy as np
from core.redis_client import redis_client
from redis.commands.search.query import Query
import json
import hashlib

class SemanticCacheRepository:

    def find_similar(self, embedding):
        vector = np.array(embedding, dtype=np.float32).tobytes()
        q = Query(
            "*=>[KNN 1 @embedding $vec AS score]"
        ).sort_by("score").return_fields("answer", "score").dialect(2)

        results = redis_client.ft("idx:cache").search(
            q,
            query_params={"vec": vector}
        )

        if results.docs:
            doc = results.docs[0]
            score = float(doc.score)
            # For cosine, 0 is identical, so use a low threshold
            if score < 0.2:
                # answer is a JSON string
                return json.loads(doc.answer)
        return None

    def save(self, embedding, query, answer):
        vector = np.array(embedding, dtype=np.float32).tobytes()
        key = f"cache:{hashlib.sha256(query.encode()).hexdigest()}"
        answer_str = json.dumps(answer)

        # Store as top-level fields
        redis_client.hset(
            key,
            mapping={
                "embedding": vector,
                "query": query,
                "answer": answer_str
            }
        )
        redis_client.expire(key, 3600)  # TTL