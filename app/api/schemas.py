from pydantic import BaseModel, Field
from typing import List, Optional

class QuestionRequest(BaseModel):
    query: str = Field(..., description="Вопрос пользователя для RAG-системы")

class AnswerResponse(BaseModel):
    answer: str = Field(..., description="Ответ, сформированный на основе найденного контекста")

class HealthCheckResponse(BaseModel):
    status: str
    db_loaded: bool
    model: str

class DocumentChunkInfo(BaseModel):
    content: str
    page: int

class DebugRetrievalResponse(BaseModel):
    query: str
    chunks: List[DocumentChunkInfo]