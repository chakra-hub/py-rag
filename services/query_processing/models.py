from pydantic import BaseModel, Field
from typing import Any
from .enums import QueryIntent, Severity


class ValidationResult(BaseModel):
    valid:bool
    errors: list[str] = Field(default_factory=list)

class GuardrailResult(BaseModel):
    blocked: bool = False
    rule: str | None = None
    severity: Severity | None = None
    reason: str | None = None
    warnings: list[str] = Field(default_factory=list)

class ProcessedQuery(BaseException):
    original_query:str
    normalized_query: str | None = None
    rewritten_query:str | None = None
    intent: QueryIntent | None = None
    metadata_filters : dict[str, Any]= Field(default_factory=dict)
    estimated_tokens: int = 0
    warnings: list[str] = Field(default_factory=dict)
    estimated_tokens: int = 0
    warnings: list[str] = Field(default_factory=list)

class MetadataFilters(BaseModel):
    collection_name: str | None = None
    filename: str | None = None
    headings: list[str] = []
    page_numbers: list[int] = []