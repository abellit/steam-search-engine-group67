# Steam Search Engine - Group 67

A field-weighted search engine for the Steam games dataset using BM25F and neural re-ranking.

## Tech Stack
- Elasticsearch 8.13.0 (via Docker) - BM25F indexing and retrieval
- sentence-transformers (all-MiniLM-L6-v2) - neural re-ranking stage
- scikit-learn - TF-IDF baseline
- NLTK - text preprocessing (tokenisation, stopword removal, stemming)
- Streamlit - search interface
- ir-measures - evaluation metrics (MAP, nDCG, MRR, P@10)

## Prerequisites
- Python 3.11+
- Docker Desktop (must be running before starting Elasticsearch)
  Download: https://www.docker.com/products/docker-desktop/
- Git

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/abellit/steam-search-engine-group67.git
cd steam-search-engine-group67
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK data
```bash
python scripts/download_nltk_data.py
```

### 5. Set up environment variables
```bash
# Windows
copy .env.example .env

# Mac/Linux
cp .env.example .env
```
No changes needed — default values work out of the box with Docker.

### 6. Start Elasticsearch
Make sure Docker Desktop is open and running, then:
```bash
docker compose up -d
```

Verify it is running:
```bash
curl http://localhost:9200
```
You should see a JSON response containing "You Know, for Search".

### 7. Build the index

Using the included 500-game sample (recommended for verification):
```bash
python scripts/build_index.py
```

Using the full dataset (~83,000 games) for the complete demo:
```bash
# First download games_may2024_cleaned.csv from:
# https://www.kaggle.com/datasets/artermiloff/steam-games-dataset
# Place it in data/cleaned_data/ folder then run:
python scripts/build_index.py --data data/cleaned_data/games_may2024_cleaned.csv
```

### 8. Run the demo
```bash
streamlit run app/streamlit_app.py
```

### 9. Run evaluation
```bash
python scripts/run_evaluation.py
```

## Stopping Elasticsearch
```bash
# Stop but keep index data
docker compose down

# Stop and delete all index data (full reset)
docker compose down -v
```

## Dataset
A 500-game sample is included in `data/sample_data/` and is sufficient enough to run and verify the full system. The full dataset is not committed due 
to its size (422MB). See step 7 above to download and use it.

## Team - Group 67
- [Name 1] - preprocessing and indexing
- Abel - retrieval and re-ranking
- [Name 2] - evaluation and GUI