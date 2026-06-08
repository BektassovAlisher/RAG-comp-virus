from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from app import config
from app.core.fusion import HybridEnsembleRetriever

class RAGPipeline:
  
    def __init__(self, rebuild_db: bool = False):
        
        self.retriever = HybridEnsembleRetriever(rebuild_db=rebuild_db)
        
        self.llm = OllamaLLM(
            model=config.LLM_MODEL,
            temperature=0.0,
            base_url=config.OLLAMA_BASE_URL
        )

        self.prompt = ChatPromptTemplate.from_template(self._get_template())
        
        self.chain = (
            {
                "context": RunnableLambda(self.retriever.retrieve),
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _get_template(self) -> str:
        return """Ты — эксперт по информационной безопасности. Отвечай на вопросы только на основе предоставленного текста.

ПРАВИЛА:
1. Используй ТОЛЬКО предоставленный КОНТЕКСТ.
2. Если в тексте нет прямого ответа, напиши: "В предоставленных фрагментах информация не найдена".
3. В конце ответа ОБЯЗАТЕЛЬНО укажи источник (номер страницы), который ты взял из текста.

КОНТЕКСТ:
{context}

ВОПРОС: {question}

ОТВЕТ:"""

    def ask(self, query: str) -> str:
      
        return self.chain.invoke(query)

_pipeline_instance = None

def get_rag_pipeline(rebuild_db: bool = False) -> RAGPipeline:
   
    global _pipeline_instance
    if _pipeline_instance is None or rebuild_db:
        _pipeline_instance = RAGPipeline(rebuild_db=rebuild_db)
    return _pipeline_instance