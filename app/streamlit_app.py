# Importing relevant libraries
import os
import sys
from pathlib import Path

# Avoids window library conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add repo root so Python can find src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd # pandas to build metric table
import streamlit as st # GUI library
import ir_measures # Evaluation libary
from src.retrieval import search_tfidf, search_bm25, search_bm25f # Retrieval models
from ir_measures import MAP, MRR, nDCG # Metrics that we show

#Sets page title
st.set_page_config(page_title="Steam Search Engine")

# Main page title
st.title("Steam Game Search Engine")
st.write("Search for games and view the ranked results.")

# User inputs query
query = st.text_input("Enter your query")

# Dropdown to choose models
model_choice = st.selectbox(
    "Choose a retrieval model",
    ["TF-IDF", "BM25", "BM25F", "BM25F + Reranker"]
)

# Run each of the models
def run_search(query_text, model_name):
    if model_name == "TF-IDF":
        return search_tfidf(query_text, top_k=10)

    if model_name == "BM25":
        return search_bm25(query_text, top_k=10)

    if model_name == "BM25F":
        return search_bm25f(query_text, top_k=10)

    if model_name == "BM25F + Reranker":
        from src.reranker import rerank_documents
        candidates = search_bm25f(query_text, top_k=20)

        return rerank_documents(query_text, candidates, top_k=10)

    return []

# Only search if the user has typed
if query.strip():
    results = run_search(query, model_choice)

    st.subheader(f"Results for: {query}")
    st.caption(f"Model used: {model_choice}")
    
    # Show ranked results
    if results:
        for i, result in enumerate(results, start=1):
            title = result.get("title", "No title")
            app_id = result.get("app_id", "N/A")

            if model_choice == "BM25F + Reranker":
                score = result.get("reranked_score", 0.0)
            else:
                score = result.get("score", 0.0)
                

            st.markdown(f"{i}. {title}")
            st.write(f"App ID: {app_id}")
            st.write(f"Score: {score:.4f}")
            st.divider()
    else:
        st.warning("No results found.")
else:
    st.info("Type a query to start searching.")


#Load queries from file
def load_queries(path):
    queries = {}

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            query = line.strip()
            if query:
                queries[str(i)] = query

    return queries

# Relevance from the qrels file
def load_qrels(path):
    qrels = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()
            # Trec qrels should only have 4 columns
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

# Run a model over all the test quiries
def make_run(search_fn, queries, top_k):
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

# Run BM25f and then rerank
def bm25f_rerank_search(query_text, top_k: int = 10):
    from src.reranker import rerank_documents
    candidates = search_bm25f(query_text, top_k=20)
    return rerank_documents(query_text, candidates, top_k=top_k)


@st.cache_data
# Build a table of all the eval metrics
def build_metrics_table():
    queries_path = ROOT / "queries" / "test_queries.txt"
    qrels_path = ROOT / "queries" / "qrels.txt"

    # Load eval data
    queries = load_queries(queries_path)
    qrels = load_qrels(qrels_path)

    metrics = [MAP@10, nDCG@10, MRR@10]


    tfidf_run = make_run(search_tfidf, queries, top_k=10)
    bm25_run = make_run(search_bm25, queries, top_k=10)
    bm25f_run = make_run(search_bm25f, queries, top_k=10)
    bm25f_rerank_run = make_run(bm25f_rerank_search, queries, top_k=10)

    # Calculate metric scores
    tfidf_results = ir_measures.calc_aggregate(metrics, qrels, tfidf_run)
    bm25_results = ir_measures.calc_aggregate(metrics, qrels, bm25_run)
    bm25f_results = ir_measures.calc_aggregate(metrics, qrels, bm25f_run)
    bm25f_rerank_results = ir_measures.calc_aggregate(metrics, qrels, bm25f_rerank_run)

    # Put scores into a table
    results_df = pd.DataFrame([
        {
            "Model": "TF-IDF",
            "MAP@10": tfidf_results[MAP@10],
            "nDCG@10": tfidf_results[nDCG@10],
            "MRR@10": tfidf_results[MRR@10],
        },
        {
            "Model": "BM25",
            "MAP@10": bm25_results[MAP@10],
            "nDCG@10": bm25_results[nDCG@10],
            "MRR@10": bm25_results[MRR@10],
        },
        {
            "Model": "BM25F",
            "MAP@10": bm25f_results[MAP@10],
            "nDCG@10": bm25f_results[nDCG@10],
            "MRR@10": bm25f_results[MRR@10],
        },
        {
            "Model": "BM25F + Reranker",
            "MAP@10": bm25f_rerank_results[MAP@10],
            "nDCG@10": bm25f_rerank_results[nDCG@10],
            "MRR@10": bm25f_rerank_results[MRR@10],
        }
    ])

    return results_df.round(4)

# Section for the eval metrics
st.subheader("Evaluation Metrics")

# Button to show table when metric
if st.button("Show metrics table"):
    try:
        metrics_df = build_metrics_table()
        st.dataframe(metrics_df, use_container_width=True)
    except Exception as e:
        st.error(f"Could not build metrics table: {e}")