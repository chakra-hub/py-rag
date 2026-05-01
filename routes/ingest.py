from fastapi import APIRouter, UploadFile, File, Form
from typing import Annotated
import shutil
from services.ingest_service import IngestService

router = APIRouter()

ingest_service=IngestService()
@router.post('/ingest')
async def ingest_doc(file: Annotated[UploadFile, File(...)], description: Annotated[str, Form()]=None):
    file_path = f"uploads/{file.filename}"
    with open(file_path,"wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        ingest_service.ingest_document(file_path)
        return {"filename": file.filename, "description": description, "status": "success", "message": "Document ingested successfully"}
    except Exception as e:
        return {"filename": file.filename, "status": "error", "message": str(e)}
