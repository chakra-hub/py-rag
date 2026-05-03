from redis.commands.search.field import VectorField, TextField
from redis.commands.search.index_definition import IndexDefinition
from core.redis_client import redis_client

def create_index():
    try:
        redis_client.ft("idx:cache").dropindex()
    except:
        pass
    redis_client.ft("idx:cache").create_index(
        [
            TextField("query"),
            VectorField(
                "embedding",
                "FLAT", {
                    "TYPE": "FLOAT32",
                    "DIM": 384, 
                    "DISTANCE_METRIC": "COSINE"
                }
            )
        ],
        definition=IndexDefinition(prefix=["cache:"])
    )

create_index()