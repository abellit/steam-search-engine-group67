import argparse
import pandas as pd
from dotenv import load_dotenv
from elasticsearch.helpers import bulk
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.retrieval import es_client

load_dotenv()
INDEX_NAME = os.getenv("ES_INDEX")

def create_index(recreate=False):
    
    # Doing a safety check
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

    # Read the csv file
    df = pd.read_csv(filepath)

    # Replacing missing values with empty strings
    df = df.fillna("")

    # Printing the number of documents we are indexing
    print(f"Indexing {len(df)} documents...")

    actions = [
        {
            "_index": INDEX_NAME,
            "_id": row["AppID"],
            "_source": {
                "app_id": row["AppID"],
                "title": row["name"],
                "short_description": row["short_description"],
                "detailed_description": row["detailed_description"],
                "genres": row["genres"],
                "tags": row["tags"]
            }
        }
        for _, row in df.iterrows()
    ]

    bulk(
        es_client, 
        actions,
        chunk_size=250,
        request_timeout=60
    )
    print(f"{len(df)} documents have been indexed.")
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="data/sample_data/games_sample.csv",
        help="Path to the CSV dataset"
    )
    args = parser.parse_args()

    create_index()
    index_documents_bm25(args.data)
