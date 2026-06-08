import os
import shutil
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from app import config
from app.core.document_processor import load_and_split_pdf

embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDINGS_MODEL)

def get_vectorstore(rebuild: bool = False) -> Chroma:
    db_exists = os.path.exists(config.CHROMA_PATH) and len(os.listdir(config.CHROMA_PATH)) > 0
    
    if rebuild or not db_exists:
        print(f"🔄 Rebuilding Chroma vector database at {config.CHROMA_PATH}...")
        if os.path.exists(config.CHROMA_PATH):
            shutil.rmtree(config.CHROMA_PATH)
            
        docs = load_and_split_pdf()
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=config.CHROMA_PATH
        )
        print("✅ Vector database built successfully.")
    else:
        print(f"📁 Loading existing vector database from {config.CHROMA_PATH}...")
        vectorstore = Chroma(
            persist_directory=config.CHROMA_PATH,
            embedding_function=embeddings
        )
    return vectorstore