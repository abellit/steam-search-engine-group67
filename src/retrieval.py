from elasticsearch import Elasticsearch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing import preprocess_text
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

host = os.getenv("ES_HOST")
port = os.getenv("ES_PORT")
INDEX_NAME = os.getenv("ES_INDEX")

es_client = Elasticsearch(host + ":" + port)

# Establishing a connection to elasticsearch built within a docker container
try:
    info = es_client.info()
    print(f"Connected to Elasticsearch: {info['version']['number']}")
except Exception as e:
    print(f"Could not connect to Elasticsearch: {e}")
    raise


# Mapping each key field
FIELD_MAP = {
    "title": "name",
    "short_description": "short_description",
    "detailed_description": "detailed_description",
    "genres": "genres",
    "tags": "tags"
}

# Creating field values
FIELD_WEIGHTS = {
    "title": 3.5,
    "short_description": 2.5,
    "genres": 3.0,
    "detailed_description": 1.5,
    "tags": 2.5
}

# Reading CSV data and filltering out non applicable values from the dataset
df = pd.read_csv("data/sample_data/games_sample.csv").fillna("")

# Combnie fields into one text representation per document
corpus = df.apply(
    lambda row: " ".join(
        preprocess_text(str(row["name"])) +
        preprocess_text(str(row["short_description"])) +
        preprocess_text(str(row["genres"])) +
        preprocess_text(str(row["tags"]))
    ), axis=1
)

vectoriser = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectoriser.fit_transform(corpus)

def search_tfidf(query: str, top_k: int = 10) -> list[dict]:
    # Search using TF-IDF across all fields equally.
    query = " ".join(preprocess_text(query))
    query_vector = vectoriser.transform([query])
    scores = cosine_similarity(query_vector, tfidf_matrix)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "app_id": str(df.iloc[idx]["AppID"]),
                "title": df.iloc[idx]["name"],
                "short_description": df.iloc[idx]["short_description"],
                "score": float(scores[idx])
            })

    return results


def search_bm25(query: str, top_k: int = 10) -> list[dict]:
    # Search using flat BM25 across all fields equally.
    query = " ".join(preprocess_text(query))
    response = es_client.search(
        index=INDEX_NAME,
        query={
            "multi_match": {
                "query": query,
                "fields": list(FIELD_WEIGHTS.keys()),
                "type": "best_fields"
            }
        },
        size=top_k
    )

    return _parse_results(response)


def search_bm25f(query: str, top_k: int = 10) -> list[dict]:
    # Search using BM25F with field weights.
    query = " ".join(preprocess_text(query))
    # Build boosted fields list from FIELD_WEIGHTS
    boosted_fields = [f"{field}^{weight}" for field, weight in FIELD_WEIGHTS.items()]

    response = es_client.search(
        index=INDEX_NAME,
        query={
            "multi_match": {
                "query": query,
                "fields": boosted_fields,
                "type": "best_fields"
            }
        },
        size=top_k
    )

    return _parse_results(response)


def _parse_results(response) -> list[dict]:
    # Extract relevant fields from Elasticsearch response.
    results = []
    for hit in response["hits"]["hits"]:
        results.append({
            "app_id": hit["_source"]["app_id"],
            "title": hit["_source"]["title"],
            "short_description": hit["_source"]["short_description"],
            "score": hit["_score"]
        })

    return results