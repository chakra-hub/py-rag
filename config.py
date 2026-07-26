from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    groq_api_key: str
    chroma_api_key: str
    chroma_tenant_id: str
    langfuse_secret_key: str
    langfuse_public_key: str
    langfuse_base_url: str
    redis_host: str = "localhost"
    redis_port: int = 6379
    chroma_database: str = "py-rag"
    chroma_collection: str = "resume_dubai"
    chroma_collection_version: str = "v2"

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()