from functools import lru_cache
from sentence_transformers import SentenceTransformer, util

# Load model once at module level
@lru_cache(maxsize=1)
def get_model():
    
    return SentenceTransformer("all-MiniLM-L6-v2")


def rerank_documents(query: str, candidates: list[dict], top_k: int = 10) -> list[dict]:
    """Reramk game document candidates using semantic similarity"""
    
    # If no results return []
    if not candidates:
        return []

    # Text representation for each candidate
    candidate_texts = [
        f"{c['title']} - {c['short_description']}"
        for c in candidates
    ]

    # All candidates and query encoded into vectors
    model = get_model()
    query_embeddings = model.encode(query)
    candidate_embeddings = model.encode(candidate_texts)

    # Compute the cosine similarity between the query and each candidate
    scores = util.cos_sim(query_embeddings, candidate_embeddings)[0]

    # Sort candidates by similarity score in descending order
    scored_candidates = list(zip(candidates, scores.tolist()))
    scored_candidates.sort(key=lambda x: x[1], reverse=True)

    reranked = []
    for candidate, score in scored_candidates:
        candidate["reranked_score"] = score
        reranked.append(candidate)

    return reranked[:top_k]