import os
import re
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

PDF_PATH = "doc/virus.pdf"
CHROMA_PATH = "./chroma_db"

base_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

llm = OllamaLLM(
    model="qwen2.5:1.5b",
    temperature=0.0,
)

def normalize_text(s: str) -> str:
    s = s.replace("\t", " ")
    s = re.sub(r"[ ]{2,}", " ", s)     
    s = re.sub(r"\n{3,}", "\n\n", s)   
    return s.strip()

loader = PyPDFLoader(PDF_PATH)
raw_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
)

final_docs = []
for doc in raw_documents:
    page_num = doc.metadata.get("page", 0) + 1
    normalized_content = normalize_text(doc.page_content)
    
    splits = text_splitter.split_documents([doc])
    for split in splits:
        split.metadata["page"] = page_num
        split.page_content = f"[Источник: стр. {page_num}]\n{split.page_content}"
        final_docs.append(split)

if os.path.exists(CHROMA_PATH):
    import shutil
    shutil.rmtree(CHROMA_PATH)

vectorstore = Chroma.from_documents(
    documents=final_docs,
    embedding=base_embeddings,
    persist_directory=CHROMA_PATH
)

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
bm25_retriever = BM25Retriever.from_documents(final_docs)
bm25_retriever.k = 4

# --- RRF (Reciprocal Rank Fusion) ---
def reciprocal_rank_fusion(docs_lists: List[List[Document]], k=4, c=60) -> List[Document]:
    doc_scores = {}
    for docs in docs_lists:
        for rank, doc in enumerate(docs):
            content = doc.page_content 
            score = 1 / (rank + c)
            if content in doc_scores:
                doc_scores[content]['score'] += score
            else:
                doc_scores[content] = {'doc': doc, 'score': score}
    
    sorted_docs = sorted(doc_scores.values(), key=lambda x: x['score'], reverse=True)
    return [item['doc'] for item in sorted_docs[:k]]

def ensemble_retrieve(query: str) -> List[Document]:
    bm25_docs = bm25_retriever.invoke(query)
    vector_docs = vector_retriever.invoke(query)
    return reciprocal_rank_fusion([bm25_docs, vector_docs])

ensemble_retriever = RunnableLambda(ensemble_retrieve)
template = """Ты — эксперт по информационной безопасности. Отвечай на вопросы только на основе предоставленного текста.

ПРАВИЛА:
1. Используй ТОЛЬКО предоставленный КОНТЕКСТ.
2. Если в тексте нет прямого ответа, напиши: "В предоставленных фрагментах информация не найдена".
3. В конце ответа ОБЯЗАТЕЛЬНО укажи источник (номер страницы), который ты взял из текста.

КОНТЕКСТ:
{context}

ВОПРОС: {question}

ОТВЕТ:"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": RunnableLambda(ensemble_retrieve), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    questions = [
        "По среде обитания вирусы делятся на?",
        "По деструктивным возможностям вирусы можно разделить на:?",
        "приведи примеры антивирусных программ:"
    ]

    for q in questions:
        print(f"\n--- ВОПРОС: {q} ---")
        print(f"Ответ: {rag_chain.invoke(q)}")