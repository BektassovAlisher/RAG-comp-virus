import os
import shutil
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
from typing import List
from langchain_experimental.text_splitter import SemanticChunker
import re
#from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
#from langchain_community.document_compressors import FlashrankRerank

PDF_PATH = "doc/virus.pdf"
CHROMA_PATH = "./chroma_db"

if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

base_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

llm = OllamaLLM(
    model="qwen2.5:3b",
    temperature=0.1,
    num_ctx=4096,
)

loader = PyPDFLoader(PDF_PATH)
raw_documents = loader.load()

def normalize_text(s: str) ->str:
    s = s.replace("\t", " ")
    s = re.sub(r"[ ]{2,}", " ", s)     
    s = re.sub(r"\n{3,}", "\n\n", s)   
    return s.strip()

full_text = ""
for doc in raw_documents:
    page_num = doc.metadata.get("page", 0) + 1
  
    full_text += f" [[[PAGE_{page_num}]]] " + normalize_text(doc.page_content)

document = [Document(page_content=full_text, metadata={"source": PDF_PATH})]

semantic_splitter = SemanticChunker(
    embeddings=base_embeddings,   
    breakpoint_threshold_amount=88
)

semantic_docs = semantic_splitter.split_documents(document)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)

final_docs = []

for d in semantic_docs:
    if len(d.page_content) > 1200:
        final_docs.extend(text_splitter.split_documents([d]))
    else:
        final_docs.append(d)

# --- ИЗМЕНЕНИЕ: Восстанавливаем метаданные страниц для каждого чанка ---
current_page = 1
for doc in final_docs:
    match = re.search(r"\[\[\[PAGE_(\d+)\]\]\]", doc.page_content)
    if match:
        current_page = int(match.group(1))
    doc.metadata['page'] = current_page
    

print("Создание временного хранилища")
vectorstore = Chroma.from_documents(
    documents=final_docs,
    collection_name="viruses",
    embedding=base_embeddings,
    persist_directory=CHROMA_PATH
)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

bm25_retriever = BM25Retriever.from_documents(final_docs)
bm25_retriever.k = 6

# Ручная реализация ensemble через RRF (Reciprocal Rank Fusion)
def reciprocal_rank_fusion(docs_lists: List[List[Document]], k=6, c=60) -> List[Document]:
    """Объединяет результаты нескольких ретриверов через RRF"""
    doc_scores = {}
    
    for docs in docs_lists:
        for rank, doc in enumerate(docs):
            doc_content = doc.page_content 
            score = 1 / (rank + c)
            
            if doc_content in doc_scores:
                doc_scores[doc_content]['score'] += score
            else:
                doc_scores[doc_content] = {'doc': doc, 'score': score}
    
    sorted_docs = sorted(doc_scores.values(), key=lambda x: x['score'], reverse=True)
    return [item['doc'] for item in sorted_docs[:k]]

# Создаем ensemble retriever через RunnableParallel
def ensemble_retrieve(query: str) -> List[Document]:
    bm25_docs = bm25_retriever.invoke(query)
    vector_docs = vector_retriever.invoke(query)
    return reciprocal_rank_fusion([bm25_docs, vector_docs], k=6)

ensemble_retriever = RunnableLambda(ensemble_retrieve)

def format_docs(docs):
    if not docs:
        return "Контекст пуст"
    formatted = []
    for i, d in enumerate(docs):
        # Теперь metadata['page'] существует, так как мы её восстановили выше
        page_num = d.metadata.get("page", "?")
        formatted.append(f"--- ФРАГМЕНТ №{i+1} (Страница {page_num}) ---\n{d.page_content}")
    return "\n\n".join(formatted)


#compressor = FlashrankRerank(model="ms-marco-MultiBERT-L-12", top_n = 3)
#ontextual_retirever = ContextualCompressionRetriever(
    #base_compressor = compressor,
    #base_retriever = ensemble_retriever
#)
template = """Ты — эксперт по информационной безопасности. Твоя задача — отвечать на вопросы по лекции «Компьютерные вирусы и антивирусные программы».

ПРАВИЛА:
1. Отвечай ТОЛЬКО на основе КОНТЕКСТА.
2. Если информации нет — ответь: "В предоставленных фрагментах информация не найдена".
3. ОТВЕТ (строго на русском, обязательно укажи страницу источника):
4. Забудь всё, что ты знаешь о вирусах из интернета. Используй только предоставленный текст".
5. Будь точным и кратким.

КОНТЕКСТ:
{context}

ВОПРОС: {question}

ОТВЕТ (строго на русском, со ссылкой на источник):"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    questions = [
        "По среде обитания вирусы делятся на?",
        "что такое загрузочный вирус?",
        "когда появились первые конструкторы вирусов?",
        "По деструктивным возможностям вирусы можно разделить на:?",
        "что такок Утилиты скрытого администрирования?",
        "приведи примеры антивирусных программ:"
    ]

    for q in questions:
        print(f"\n--- ПОИСК ДЛЯ ВОПРОСА: {q} ---")
        response = rag_chain.invoke(q)
        print(f"Ответ: {response}")
        print("-" * 60)