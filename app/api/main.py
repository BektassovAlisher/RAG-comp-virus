import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from app import config
from app.api.schemas import QuestionRequest, AnswerResponse, HealthCheckResponse, DebugRetrievalResponse, DocumentChunkInfo
from app.core.rag import get_rag_pipeline
from app.core.database import get_vectorstore

app = FastAPI(
    title="🛡️ RAG Security Advisor API",
    description="FastAPI backend for retrieval-augmented generation on Computer Viruses and Security.",
    version="1.0.0"
)

is_rebuilding = False

@app.on_event("startup")
def startup_event():
    """
    Prefetch the pipeline on startup to load embeddings and index.
    """
    try:
        get_rag_pipeline()
    except Exception as e:
        print(f"⚠️ Warning during startup pipeline initialization: {e}")

@app.get("/health", response_model=HealthCheckResponse)
def health_check():
    """
    Check API and vector store status.
    """
    db_loaded = os.path.exists(config.CHROMA_PATH) and len(os.listdir(config.CHROMA_PATH)) > 0
    return HealthCheckResponse(
        status="rebuilding" if is_rebuilding else "ok",
        db_loaded=db_loaded,
        model=config.LLM_MODEL
    )

@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):

    if is_rebuilding:
        raise HTTPException(status_code=503, detail="База данных в процессе обновления. Пожалуйста, подождите.")
        
    try:
        pipeline = get_rag_pipeline()
        answer = pipeline.ask(request.query)
        return AnswerResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки RAG: {str(e)}")

def rebuild_db_task():
    global is_rebuilding
    try:
        is_rebuilding = True
        get_rag_pipeline(rebuild_db=True)
    finally:
        is_rebuilding = False

@app.post("/rebuild")
def rebuild_database(background_tasks: BackgroundTasks):
    global is_rebuilding
    if is_rebuilding:
        return {"status": "already_rebuilding", "message": "Процесс пересборки уже запущен."}
        
    background_tasks.add_task(rebuild_db_task)
    return {"status": "started", "message": "Пересборка базы данных запущена в фоновом режиме."}

@app.post("/debug/retrieve", response_model=DebugRetrievalResponse)
def debug_retrieve(request: QuestionRequest):

    try:
        pipeline = get_rag_pipeline()
        docs = pipeline.retriever.retrieve(request.query)
        chunks = [
            DocumentChunkInfo(content=doc.page_content, page=doc.metadata.get("page", 1))
            for doc in docs
        ]
        return DebugRetrievalResponse(query=request.query, chunks=chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api.main:app", host=config.API_HOST, port=config.API_PORT, reload=True)