import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = os.getenv("PDF_PATH", str(BASE_DIR / "doc" / "virus.pdf"))
CHROMA_PATH = os.getenv("CHROMA_PATH", str(BASE_DIR / "chroma_db"))

LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b ")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 4
RRF_C = 60  

API_HOST = "0.0.0.0"
API_PORT = 8000
API_URL = os.getenv("API_URL", f"http://localhost:{API_PORT}")

EVAL_LLM_MODEL = "qwen2.5:3b"
EVAL_RESULTS_PATH = str(BASE_DIR / "ragas_results.csv")
