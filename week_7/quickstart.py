"""Minimal end-to-end smoke test of the baseline RAG pipeline.

Run this AFTER:
    1. pip install -r requirements.txt
    2. cp .env.example .env  and paste your Gemini key into .env

    python quickstart.py

It builds a tiny subset, indexes it, and answers one benchmark question so you
can confirm retrieval + generation work end to end before running the full
notebook.
"""
from src import dataset
from src.pipeline import RAGPipeline


def main() -> None:
    print("1) Loading a tiny open_ragbench subset (downloads on first run)...")
    docs, examples = dataset.load_subset(sample_size=5, n_negatives=10)
    print(f"   -> {len(docs)} documents, {len(examples)} questions\n")

    print("2) Indexing (chunk -> embed -> store)...")
    pipe = RAGPipeline(collection_name="quickstart", reset=True)
    n_chunks = pipe.index_documents(dataset.records(docs))
    print(f"   -> indexed {n_chunks} chunks\n")

    example = examples[0]
    print("3) Answering a ground-truth question...")
    print(f"   Q: {example.question}\n")
    result = pipe.answer(example.question)
    print(f"   A: {result.answer}\n")

    retrieved_docs = [c.doc_id for c in result.contexts]
    hit = example.gold_doc_id in retrieved_docs
    print(f"   Retrieved from docs: {retrieved_docs}")
    print(f"   Gold doc: {example.gold_doc_id}  ->  {'HIT ✅' if hit else 'miss ❌'}")
    print(f"\n   Reference answer (truncated): {example.reference_answer[:300]}")
    print("\nDone. If you see an answer and a HIT, the baseline pipeline works.")


if __name__ == "__main__":
    main()
