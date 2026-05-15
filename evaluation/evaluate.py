import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq
from evaluation.test_dataset import test_cases
from repository.vector_database import VectorRepository
from repository.bm25_repository import BM25Repository
from services.ask_llm import AskLLM
import uuid
from langfuse.langchain import CallbackHandler

# Setup
vector_db = VectorRepository()
bm25_db = BM25Repository()
ask_llm = AskLLM()

async def run_rag(question: str) -> dict:
    """Run your RAG pipeline and return answer + contexts"""
    # Hybrid retrieval
    vector_results = vector_db.query_text(question)[:2]
    bm25_results = bm25_db.query(question, n_results=2)

    seen = set()
    chunks = []
    for chunk in vector_results + bm25_results:
        if chunk not in seen:
            seen.add(chunk)
            chunks.append(chunk)
    chunks = chunks[:3]

    if not chunks:
        return {
            "answer": "I cannot find this information in the document",
            "contexts": []
        }
    
    context_text = "\n\n---\n\n".join(chunks)
    handler = CallbackHandler()
    result = await ask_llm.chat_with_groq(
        session_id=str(uuid.uuid4()),
        question=question,
        context=context_text,
        callbacks=[handler]
    )

    return {
        "answer": result["response"],
        "contexts": chunks
    }

async def build_evaluation_dataset():
    """Run all test cases through RAG and collect results"""
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print(f"Running {len(test_cases)} test cases...")

    for i, case in enumerate(test_cases):
        print(f"[{i+1}/{len(test_cases)}] {case['question'][:50]}...")
        result = await run_rag(case["question"])
        questions.append(case["question"])
        answers.append(result["answer"])
        contexts.append(result["contexts"])
        ground_truths.append(case["ground_truth"])

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

async def main():
    # Build dataset
    dataset = await build_evaluation_dataset()
    model = ChatGroq(
            groq_api_key=settings.groq_api_key, model_name="llama-3.3-70b-versatile"
        )
    wrapped_llm = LangchainLLMWrapper(model)

    # Run evaluation
    print("\nRunning RAGAS evaluation...")
    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=wrapped_llm
    )

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    df = results.to_pandas()
    print(f"Faithfulness:      {df['faithfulness'].mean():.3f}")
    print(f"Answer Relevancy:  {df['answer_relevancy'].mean():.3f}")
    print(f"Context Precision: {df['context_precision'].mean():.3f}")
    print(f"Context Recall:    {df['context_recall'].mean():.3f}")
    print("="*50)

    # Show worst performing questions
    print("\nWORST PERFORMING (Faithfulness < 0.7):")
    low_faith = df[df['faithfulness'] < 0.7]
    for _, row in low_faith.iterrows():
        print(f"  Q: {row['question'][:60]}")
        print(f"  Faithfulness: {row['faithfulness']:.3f}")
        print()

    # Save full results
    df.to_csv("evaluation/results.csv", index=False)
    print("Full results saved to evaluation/results.csv")

if __name__ == "__main__":
    asyncio.run(main())