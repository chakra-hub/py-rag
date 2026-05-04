from fastapi import APIRouter
from repository.semantic_cache import SemanticCacheRepository

router = APIRouter()
cache_repo = SemanticCacheRepository()

@router.delete("/cache")
def clear_cache():
    try:
        result = cache_repo.clear_all()
        return {"status": "success", "message": f"Cleared {result} cache entries"}
    except Exception as e:
        raise Exception(f"Failed to clear cache: {str(e)}")