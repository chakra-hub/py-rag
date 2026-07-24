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

    database_name: Annotated[str, Form(...)] = ...,
    collection_name: Annotated[str, Form(...)] = ...,
    version: Annotated[str, Form(...)] = ...,

    description: Annotated[str | None, Form()] = None,
):

    if file is None and not url:
        raise HTTPException(
            status_code=400,
            detail="Either file or url must be provided."
        )

    if file is not None and url:
        raise HTTPException(
            status_code=400,
            detail="Provide either file or url, not both."
        )

    if file is not None:

        os.makedirs("uploads", exist_ok=True)

        file_path = f"uploads/{file.filename}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        source = file_path

    else:
        source = url

    return ingest_service.ingest_document(
        uploaded_file=source,
        database_name=database_name,
        collection_name=collection_name,
        version=version,
    )