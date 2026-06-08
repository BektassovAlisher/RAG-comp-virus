import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import OllamaLLM
from app import config
from app.core.rag import get_rag_pipeline
from app.core.database import embeddings

def run_evaluation():
    print("🤖 Initializing RAG Pipeline and Evaluator LLM...")
    pipeline = get_rag_pipeline()
    
    evaluator_llm = OllamaLLM(
        model=config.EVAL_LLM_MODEL, 
        temperature=0.0,
        base_url=config.OLLAMA_BASE_URL
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)
    
    questions = [
        "какие есть виды вирусоподобных программ?",
        "из скольки частей состоит антивирусная программа? и объясни их",
        "По среде обитания вирусы делятся на?",
        "что такое загрузочный вирус?",
        "когда появились первые конструкторы вирусов?"
    ]
    
    ground_truths = [
        "троянские программы, утилиты скрытого администрирования, intended-вирусы, конструкторы вирусов, полиморфик-генераторы",
        "Модуль резидентной защиты, Модуль карантина, Модуль протектора, Коннектор, Модуль обновления, Модуль сканера",
        "файловые, загрузочные, макровирусы, сетевые",
        "Загрузочные вирусы записывают себя либо в загрузочный сектор диска (boot-сектор), либо в сектор, содержащий системный загрузчик жесткого диска (Master Boot Record), либо меняют указатель на активный boot-сектор",
        "В 1992 году появились первые конструкторы вирусов VCL и PS-MPC, которые увеличили и без того немаленький поток новых вирусов."
    ]
    
    answers = []
    contexts = []
    
    print("🔍 Fetching contexts and generating answers for evaluation...")
    for q in questions:
        print(f"-> Processing: '{q}'")
        
        docs = pipeline.retriever.retrieve(q)
        contexts.append([doc.page_content for doc in docs])
        
        answer = pipeline.ask(q)
        answers.append(answer)
    
    print("📊 Constructing dataset for Ragas...")
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })
    
    print("🚀 Running Ragas evaluation (requires qwen2.5:3b to be pulled in Ollama)...")
    try:
        results = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings
        )
        print("\n✨ Evaluation Complete! Results:")
        print(results)
        
        df = results.to_pandas()
        print("\n📈 Mean Metric Scores:")
        print(df[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']].mean())
        
        df.to_csv(config.EVAL_RESULTS_PATH, index=False)
        print(f"💾 Results saved to {config.EVAL_RESULTS_PATH}")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        print("💡 Make sure you have pulled the evaluator model: 'ollama pull qwen2.5:3b'")

if __name__ == "__main__":
    run_evaluation()