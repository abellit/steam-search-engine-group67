# Runs all three retrieval models (TF-IDF, BM25F, BM25F + reranker)
# against test_queries.txt and qrels.txt, then prints a metrics table.
# Usage: python scripts/run_evaluation.py
from pathlib import Path  # for file paths
import sys  # lets us add the project root to imports

import pandas as pd  # used for the final results table
import ir_measures  # evaluation library
from ir_measures import MAP, MRR, nDCG, P  # metrics we want to report

# Add repo root so Python can find src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.reranker import rerank_documents  # neural reranker
from src.retrieval import search_tfidf, search_bm25, search_bm25f  # retrieval models


def load_queries(path: Path) -> dict[str, str]:
    """Load test queries and give each one an id."""
    queries = {}

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            query = line.strip()
            if query:
                queries[str(i)] = query

    return queries


def load_qrels(path: Path):
    """Load relevance judgements from qrels file."""
    qrels = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Skip blank lines and comments
            if not line or line.startswith("#"):
                continue

            parts = line.split()

            # Skip anything not in standard 4-column format
            if len(parts) != 4:
                continue

            query_id, _, doc_id, relevance = parts

            qrels.append(
                ir_measures.Qrel(
                    query_id=query_id,
                    doc_id=doc_id,
                    relevance=int(relevance)
                )
            )

    return qrels


def make_run(search_fn, queries: dict[str, str], top_k: int = 10):
    """Run one retrieval model for all queries."""
    run = []

    for query_id, query_text in queries.items():
        results = search_fn(query_text, top_k=top_k)

        for doc in results:
            score = float(doc.get("score", doc.get("reranked_score", 0.0)))

            run.append(
                ir_measures.ScoredDoc(
                    query_id=query_id,
                    doc_id=str(doc["app_id"]),
                    score=score
                )
            )

    return run

# Run BM25F and then rerank results
def bm25f_rerank_search(query: str, top_k: int = 10):
    """Run BM25F first, then rerank the top results."""
    candidates = search_bm25f(query, top_k=20)
    reranked = rerank_documents(query, candidates, top_k=top_k)
    return reranked

# Calculate the evaluation score for a model
def evaluate_model(run, qrels, metrics):
    return ir_measures.calc_aggregate(metrics, qrels, run)


def main():
    # File paths
    queries_path = ROOT / "queries" / "test_queries.txt"
    qrels_path = ROOT / "queries" / "qrels.txt"

    # Load test data
    queries = load_queries(queries_path)
    qrels = load_qrels(qrels_path)

    # Metrics to report
    metrics = [MAP@10, nDCG@10, MRR@10, P@10]

    # Build runs for each model
    tfidf_run = make_run(search_tfidf, queries, top_k=10)
    bm25_run = make_run(search_bm25, queries, top_k=10)
    bm25f_run = make_run(search_bm25f, queries, top_k=10)
    bm25f_rerank_run = make_run(bm25f_rerank_search, queries, top_k=10)

    # Evaluate each model
    tfidf_results = evaluate_model(tfidf_run, qrels, metrics)
    bm25_results = evaluate_model(bm25_run, qrels, metrics)
    bm25f_results = evaluate_model(bm25f_run, qrels, metrics)
    bm25f_rerank_results = evaluate_model(bm25f_rerank_run, qrels, metrics)

    # Put results into a dataframe
    results_df = pd.DataFrame([
        {
            "Model": "TF-IDF",
            "MAP@10": tfidf_results[MAP@10],
            "nDCG@10": tfidf_results[nDCG@10],
            "MRR@10": tfidf_results[MRR@10],
            "P@10": tfidf_results[P@10],
        },
        {
            "Model": "BM25",
            "MAP@10": bm25_results[MAP@10],
            "nDCG@10": bm25_results[nDCG@10],
            "MRR@10": bm25_results[MRR@10],
            "P@10": bm25_results[P@10],
        },
        {
            "Model": "BM25F",
            "MAP@10": bm25f_results[MAP@10],
            "nDCG@10": bm25f_results[nDCG@10],
            "MRR@10": bm25f_results[MRR@10],
            "P@10": bm25f_results[P@10],
        },
        {
            "Model": "BM25F + Reranker",
            "MAP@10": bm25f_rerank_results[MAP@10],
            "nDCG@10": bm25f_rerank_results[nDCG@10],
            "MRR@10": bm25f_rerank_results[MRR@10],
            "P@10": bm25f_rerank_results[P@10],
        }
    ])

    # Round values for cleaner output
    results_df = results_df.round(4)

    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()