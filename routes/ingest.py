from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Annotated
import shutil
import os

from services.ingest_service import IngestService

router = APIRouter()

ingest_service = IngestService()


@router.post("/ingest")
async def ingest(
    file: Annotated[UploadFile | None, File()] = None,
    url: Annotated[str | None, Form()] = None,
    description: Annotated[str | None, Form()] = None,
):
    """
    Ingest a document from either:
    - File upload
    - URL (public webpage / Confluence URL)
    """

    # Validation
    if file is None and not url:
        raise HTTPException(
            status_code=400,
            detail="Either a file or a URL must be provided."
        )

    if file is not None and url:
        raise HTTPException(
            status_code=400,
            detail="Provide either a file or a URL, not both."
        )

    try:
        if file is not None:

            os.makedirs("uploads", exist_ok=True)

            file_path = f"uploads/{file.filename}"

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            source = file_path
            source_name = file.filename

        else:
            source = url
            source_name = url

        ingest_service.ingest_document(source)

        return {
            "status": "success",
            "source": source_name,
            "description": description,
            "message": "Document ingested successfully"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )