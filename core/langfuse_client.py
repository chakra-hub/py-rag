from langfuse import Langfuse
from config import settings

langfuse_client = Langfuse(
    secret_key=settings.langfuse_secret_key,
    public_key=settings.langfuse_public_key,
    base_url=settings.langfuse_base_url,
)
