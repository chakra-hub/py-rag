from fastapi import FastAPI
from core.create_index import create_index
from routes import ingest, chat
import uvicorn
from contextlib import asynccontextmanager

app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_index()
    print("Redis index created")
    yield
    
app.include_router(ingest.router, prefix="/api/v1", tags=['ingest'])
app.include_router(chat.router, prefix="/api/v1", tags=['chat'])

@app.get("/health")
def health():
    return {"status": "ok", "service": "py-rag"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True, reload_excludes=["uploads/*"])

