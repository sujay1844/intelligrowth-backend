from pydantic import BaseModel

class QAApiBody(BaseModel):
    n: int = 5
    topics: list = []

class FeedbackApiBody(BaseModel):
    question: str
    response: str
    expected: str