# One-time script to build the Elasticsearch index on Elastic Cloud.
# Run this only once — the index persists on the cloud.
# Usage: python scripts/build_index.py --data data/sample/games_sample.csv

import pandas as pd
from dotenv import load_dotenv
import os
from src.retrieval import es_client

load_dotenv()
INDEX_NAME = os.getenv("ES_INDEX")
TFIDF_INDEX_NAME = os.getenv("ES_INDEX_TFIDF")
file = "data\sample_data\games_sample.csv"

def create_index(recreate=False):
    """Define the index mapping and create it in Elasticsearch"""

    # Safety check
    if es_client.indices.exists(index=INDEX_NAME):
        if recreate:
            print(f"Deleting existing index '{INDEX_NAME}'...")
            es_client.indices.delete(index=INDEX_NAME)
        else:
            print(f"Index '{INDEX_NAME}' already exists. Skipping creation.")
            return

    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "similarity": {
                "title_similarity": {
                    "type": "BM25",
                    "b": 0.3,
                    "k1": 1.25
                },
                "short_description_similarity": {
                    "type": "BM25",
                    "b": 0.57,
                    "k1": 1.25
                },
                "genres_similarity": {
                    "type": "BM25",
                    "b": 0.125,
                    "k1": 1.25
                },
                "tags_similarity": {
                    "type": "BM25",
                    "b": 0.4,
                    "k1": 1.25
                },
                "detailed_description_similarity": {
                    "type": "BM25",
                    "b": 0.9,
                    "k1": 1.25
                }
            }
        },
        "mappings": {
            "properties": {
                "app_id": {"type": "keyword"},
                "title": {
                    "type": "text",
                    "analyzer": "english",
                    "similarity": "title_similarity"
                },
                "short_description": {
                    "type": "text",
                    "analyzer": "english",
                    "similarity": "short_description_similarity"
                },
                "detailed_description": {
                    "type": "text",
                    "analyzer": "english",
                    "similarity": "detailed_description_similarity"
                },
                "genres": {
                    "type": "text",
                    "analyzer": "english",
                    "similarity": "genres_similarity"
                },
                "tags": {
                    "type": "text",
                    "analyzer": "english",
                    "similarity": "tags_similarity"
                }
            }
        }
    }

    es_client.indices.create(
        index=INDEX_NAME, 
        settings=mapping["settings"],
        mappings=mapping["mappings"]
    )
    print(f"Index '{INDEX_NAME}' created successfully.")


def index_documents_bm25(filepath: str):
    """Read the CSV and index each game as a document"""

    df = pd.read_csv(filepath)

    # Replacing missing values with empty strings
    df = df.fillna("")

    # Printing the number of documents we are indexing
    print(f"Indexing {len(df)} documents...")

    for _, row in df.iterrows():
        doc = {
            "app_id": row["AppID"],
            "title": row["name"],
            "short_description": row["short_description"],
            "detailed_description": row["detailed_description"],
            "genres": row["genres"],
            "tags": row["tags"]
        }

        es_client.index(
            index=INDEX_NAME,
            id=row["AppID"],
            document=doc
        )
    print(f"{len(df)} documents have been indexed.")
   

if __name__ == "__main__":
    create_index(True)
    index_documents_bm25(file)
   