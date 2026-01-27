from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from typing import List

PDF_PATH = "doc/virus.pdf"

base_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

llm = Ollama(
    model="qwen2.5:3b",
    temperature=0.1,
    num_ctx=4096,
)

loader = PyPDFLoader(PDF_PATH)
document = loader.load()

for doc in document:
    doc.page_content = doc.page_content.replace('\n', ' ').replace('  ', ' ')

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=150,
    separators=["\n\n", ". ", " ", ""]
)
docs = text_splitter.split_documents(document)

vectorstore = Chroma.from_documents(
    docs,
    collection_name="viruses",
    embedding=base_embeddings,
    persist_directory="./chroma_db"
)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 6

# Ручная реализация ensemble через RRF (Reciprocal Rank Fusion)
def reciprocal_rank_fusion(docs_lists: List[List[Document]], k=6, c=60) -> List[Document]:
    """Объединяет результаты нескольких ретриверов через RRF"""
    doc_scores = {}
    
    for docs in docs_lists:
        for rank, doc in enumerate(docs):
            doc_id = doc.page_content[:200]  # Используем начало текста как ID
            score = 1 / (rank + c)
            
            if doc_id in doc_scores:
                doc_scores[doc_id]['score'] += score
            else:
                doc_scores[doc_id] = {'doc': doc, 'score': score}
    
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
        page_num = d.metadata.get("page", 0) + 1
        formatted.append(f"--- ФРАГМЕНТ №{i+1} (Страница {page_num}) ---\n{d.page_content}")
    return "\n\n".join(formatted)

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
        "какие есть Виды вирусоподобных программ?",
        "из скольки частей состоит антивирусная программа? и объясни их",
        "Факторы определяющие качество антивирусных программ"
    ]

    for q in questions:
        print(f"\n--- ПОИСК ДЛЯ ВОПРОСА: {q} ---")
        response = rag_chain.invoke(q)
        print(f"Ответ: {response}")
        print("-" * 60)