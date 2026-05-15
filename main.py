from fastapi import FastAPI
from core.create_index import create_index
from routes import ingest, chat, admin
import uvicorn
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()
from langfuse import Langfuse
from config import settings

langfuse = Langfuse(
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key,
    host=settings.langfuse_base_url
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs on startup
    create_index()
    print("Redis index ready")
    yield
    # Runs on shutdown
    langfuse.flush()
    print("Langfuse flushed")

app = FastAPI(lifespan=lifespan)
app.include_router(ingest.router, prefix="/api/v1", tags=['ingest'])
app.include_router(chat.router, prefix="/api/v1", tags=['chat'])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])

@app.get("/health")
def health():
    return {"status": "ok", "service": "py-rag"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True, reload_excludes=["uploads/*"])

