from pydoc import doc

from elasticsearch import Elasticsearch
import os
from dotenv import load_dotenv

load_dotenv()

ES_HOST = os.getenv("ES_HOST")
ES_PORT = os.getenv("ES_PORT")
ES_INDEX = os.getenv("ES_INDEX")

ES_USERNAME = os.getenv("ES_USERNAME")   #for cloud
ES_PASSWORD = os.getenv("ES_PASSWORD")

from elasticsearch import Elasticsearch
import time

def get_client():
    es = Elasticsearch(
        "http://localhost:9200",
        request_timeout=30,
        retry_on_timeout=True
    )

    for i in range(30):
        try:
            if es.ping():
                print("Connected to Elasticsearch")
                return es
        except:
            pass
        time.sleep(1)

    raise Exception("Elasticsearch not ready")

def create_index(es):
    if es.indices.exists(index=ES_INDEX):
        print(f"Index '{ES_INDEX}' already exists")
        return

    mapping = {
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "short_description": {"type": "text"},
                "detailed_description": {"type": "text"},
                "tags": {"type": "text"},
                "genres": {"type": "text"},
            }
        }
    }

    es.indices.create(index=ES_INDEX, body=mapping)
    print(f"Index '{ES_INDEX}' created")


def index_documents(es, docs):
    for i, doc in enumerate(docs):
        doc_id = str(doc.get("app_id", i))  #use id from doc or fallback to index
        es.index(index=ES_INDEX, id=doc_id, document=doc)

    print(f"Indexed {len(docs)} documents")