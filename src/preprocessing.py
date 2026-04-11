import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from dotenv import load_dotenv
import os

load_dotenv()
USE_STEMMING = os.getenv("USE_STEMMING", "false").lower() == "true"  #loading env settings

STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()

def preprocess_text(text):
    # handles missing values
    if text is None:   
        return []

    # make string
    text = str(text) 

    # lowercasing
    text = text.lower() 

    # removing punctuation 
    text = re.sub(r"[^a-z0-9\s]", " ", text)   

    # tokenisation
    tokens = text.split()    

    # removing stopwords
    tokens = [t for t in tokens if t not in STOPWORDS]   

    # stemming
    if USE_STEMMING:
        tokens = [STEMMER.stem(t) for t in tokens]    

    return tokens


def preprocess_document(row):
    return {
        "app_id": row.get("AppID"),
        "title": " ".join(preprocess_text(row.get("name"))),
        "short_description": " ".join(preprocess_text(row.get("short_description"))),
        "detailed_description": " ".join(
            preprocess_text(row.get("about_the_game") or row.get("detailed_description"))
        ),
        "tags": " ".join(preprocess_text(row.get("tags"))),
        "genres": " ".join(preprocess_text(row.get("genres"))),
    }


def preprocess_dataset(df):
    processed_docs = []

    for _, row in df.iterrows():
        processed_doc = preprocess_document(row)
        processed_docs.append(processed_doc)

    return processed_docs