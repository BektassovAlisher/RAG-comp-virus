import re
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app import config

def normalize_text(s: str) -> str:
    
    s = s.replace("\t", " ")
    s = re.sub(r"[ ]{2,}", " ", s)     
    s = re.sub(r"\n{3,}", "\n\n", s)   
    return s.strip()

def load_and_split_pdf(pdf_path: str = config.PDF_PATH) -> List[Document]:
    
    loader = PyPDFLoader(pdf_path)
    raw_documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, 
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    
    final_docs = []
    for doc in raw_documents:
        page_num = doc.metadata.get("page", 0) + 1
        normalized_content = normalize_text(doc.page_content)
        
        page_doc = Document(page_content=normalized_content, metadata={"page": page_num})
        splits = text_splitter.split_documents([page_doc])
        
        for split in splits:
            split.metadata["page"] = page_num
            
            split.page_content = f"[Источник: стр. {page_num}]\n{split.page_content}"
            final_docs.append(split)
            
    return final_docs