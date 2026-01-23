from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate



PDF_PATH = "doc/virus.pdf"
base_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
llm = Ollama(
    model="qwen2.5:3b",
    temperature=0.0,
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
retriever = vectorstore.as_retriever(search_kwargs= {"k": 4})


bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 4

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, retriever],
    weights=[0.4, 0.6] 
)


def format_docs(docs):
    if not docs:
        return "Контекст пуст"
    formatted = []
    for i, d in enumerate(docs):
        # Добавляем номер страницы из метаданных PDF
        page_num = d.metadata.get("page", 0) + 1
        formatted.append(f"--- ФРАГМЕНТ №{i+1} (Страница {page_num}) ---\n{d.page_content}")
    return "\n\n".join(formatted)

template = """Ты — эксперт по информационной безопасности. Твоя задача — отвечать на вопросы по лекции «Компьютерные вирусы и антивирусные программы».

ПРАВИЛА:
1. Отвечай ТОЛЬКО на основе КОНТЕКСТА.
2. Если информации нет — ответь: "В предоставленных фрагментах информация не найдена".
3. Используй ТОЛЬКО русский язык.
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
        "Объясни Dr Web",
        "Из скольки частей состоит Антивирусная программа?"
        

    ]

    for q in questions:
        print(f"\n--- ПОИСК ДЛЯ ВОПРОСА: {q} ---")
        response = rag_chain.invoke(q)
        print(f"Ответ: {response}")
        print("-" * 60)