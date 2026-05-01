from fastapi import FastAPI
from routes import ingest, chat
import uvicorn

app = FastAPI()

app.include_router(ingest.router, prefix="/api/v1", tags=['ingest'])
app.include_router(chat.router, prefix="/api/v1", tags=['chat'])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True, reload_excludes=["uploads/*"])

