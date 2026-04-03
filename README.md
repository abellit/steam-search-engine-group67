# Steam Search Engine - Group 67

A field-weighted search engine for the Steam games dataset using BM25F and neural re-ranking.

## Tech Stack
- Elasticsearch 8.13.0 (via Docker) - BM25F indexing and retrieval
- sentence-transformers - neural re-ranking stage
- scikit-learn - TF-IDF baseline
- NLTK - text preprocessing
- Streamlit - search interface
- ir-measures - evaluation metrics (MAP, nDCG, MRR, P@10)

## Prerequisites
- Python 3.11+
- Docker Desktop (must be running before starting Elasticsearch)
  Download here if not installed: https://www.docker.com/products/docker-desktop/
- Git

## How to Setup

### 1. Clone the repository
```bash
git clone https://github.com/abellit/steam-search-engine-group67.git
cd steam-search-engine-group67
```

### 2. Switch to dev and create your own feature branch
All development work happens on feature branches, not on main.
```bash
git checkout dev

# Create your own branch (replace "yourname" with your actual name)
git checkout -b yourname/feature
git push origin yourname/feature
```

Only merge into `dev` when your feature is working and tested.
Only merge `dev` into `main` when everything is integrated and ready to submit.

### 3. Create and activate a virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Download NLTK data
```bash
python scripts/download_nltk_data.py
```

### 6. Set up environment variables
```bash
# Windows
copy .env.example .env

# Mac/Linux
cp .env.example .env
```
No changes needed — the default values work out of the box with Docker.

### 7. Start Elasticsearch
Make sure Docker Desktop is open and running, then:
```bash
docker compose up -d
```

Verify it is running by visiting http://localhost:9200 in your browser,
or by running:
```bash
curl http://localhost:9200
```
You should see a JSON response containing "You Know, for Search".

### 8. Build the index (run once per machine)
```bash
python scripts/build_index.py --data data/sample_data/sam.csv
```

### 9. Run the demo
```bash
streamlit run app/streamlit_app.py
```

### 10. Run evaluation
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

## Branch Strategy
```
main          <- final submission only, do not push directly
dev           <- integration branch, merge your feature branch here first
yourname/feature <- your personal working branch
```

Workflow for every change:
1. Work on your feature branch
2. Test it locally
3. Merge into dev and test integration
4. Only when everything works, dev gets merged into main


## Features to work on
- preprocessing
- indexing
- retrieval - Abel
- re-ranking - Abel
- evaluation
- GUI