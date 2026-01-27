import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from rag import rag_chain, ensemble_retriever, llm, base_embeddings

evaluator_llm = LangchainLLMWrapper(llm)
evaluator_embeddings = LangchainEmbeddingsWrapper(base_embeddings)

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

print("Подготова оценки...")
for q in questions:
    docs = ensemble_retriever.invoke(q)
    contexts.append([doc.page_content for doc in docs])
    answers.append(rag_chain.invoke(q))

dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})


results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings
)

print(results)

df = results.to_pandas()
print("\nСредние значения:")
print(df[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']].mean())

df.to_csv("ragas_results.csv", index=False)