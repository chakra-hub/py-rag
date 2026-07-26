from pydantic import BaseModel, Field

class CollectionInfo(BaseModel):
    collection_name: str
    version: int