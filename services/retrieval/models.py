from pydantic import BaseModel


class HeadingFilterResult(BaseModel):
    where: dict | None = None