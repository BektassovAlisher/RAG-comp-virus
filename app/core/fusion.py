from typing import List
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from app import config
from app.core.database import get_vectorstore
from app.core.document_processor import load_and_split_pdf

class HybridEnsembleRetriever:
  
    def __init__(self, rebuild_db: bool = False):
        
        self.docs = load_and_split_pdf()
        self.vectorstore = get_vectorstore(rebuild=rebuild_db)
        
        self.vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVER_K})
        
        self.bm25_retriever = BM25Retriever.from_documents(self.docs)
        self.bm25_retriever.k = config.RETRIEVER_K

    def reciprocal_rank_fusion(self, docs_lists: List[List[Document]], k: int = config.RETRIEVER_K, c: int = config.RRF_C) -> List[Document]:
        
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

    def retrieve(self, query: str) -> List[Document]:
        
        bm25_docs = self.bm25_retriever.invoke(query)
        vector_docs = self.vector_retriever.invoke(query)
        return self.reciprocal_rank_fusion([bm25_docs, vector_docs])