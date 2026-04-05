from scripts.index import get_client
import os
from dotenv import load_dotenv

load_dotenv()

ES_INDEX = os.getenv("ES_INDEX", "steam_games")

def search(query, top_k=10):
    es = get_client()

    body = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": [
                    "title",
                    "short_description",
                    "detailed_description",
                    "tags",
                    "genres"
                ]
            }
        }
    }

    res = es.search(index=ES_INDEX, body=body, size=top_k)

    results = []
    for hit in res["hits"]["hits"]:
        results.append({
            "doc_id": hit["_id"],
            "score": hit["_score"],
            "doc": hit["_source"]
        })

    return results