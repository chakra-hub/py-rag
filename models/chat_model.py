from pydantic import BaseModel

class Chat(BaseModel):
    question: str