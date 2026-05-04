from langfuse import Langfuse
from config import settings


def get_langfuse_client():
    return Langfuse(
        secret_key=settings.langfuse_secret_key,
        public_key=settings.langfuse_public_key,
        base_url=settings.langfuse_base_url,
    )