import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add repo root so Python can find src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import streamlit as st
from src.retrieval import search_tfidf, search_bm25, search_bm25f

st.set_page_config(page_title="Steam Search Engine", page_icon="🎮")

st.title("Steam Game Search Engine")
st.write("Search for games and view the ranked results.")

query = st.text_input("Enter your query")
model_choice = st.selectbox(
    "Choose a retrieval model",
    ["TF-IDF", "BM25", "BM25F", "BM25F + Reranker"]
)


def run_search(query_text: str, model_name: str):
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


if query.strip():
    results = run_search(query, model_choice)

    st.subheader(f"Results for: {query}")
    st.caption(f"Model used: {model_choice}")

    if results:
        for i, result in enumerate(results, start=1):
            title = result.get("title", "No title")
            app_id = result.get("app_id", "N/A")
            
            #score = result.get("score", result.get("reranked_score", 0.0))

            if model_choice == "BM25F + Reranker":
                score = result.get("reranked_score", 0.0)
            else:
                score = result.get("score", 0.0)
                

            st.markdown(f"**{i}. {title}**")
            st.write(f"App ID: {app_id}")
            st.write(f"Score: {score:.4f}")
            st.divider()
    else:
        st.warning("No results found.")
else:
    st.info("Type a query to start searching.")