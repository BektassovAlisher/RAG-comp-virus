from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag import rag_chain 
app = FastAPI(title="RAG API")


class Question(BaseModel):
    query: str


class Answer(BaseModel):
    answer: str


@app.post("/ask", response_model=Answer)
def ask(request: Question):
    
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG система не инициализирована")
    
    try:
        answer = rag_chain.invoke(request.query)
        return Answer(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)