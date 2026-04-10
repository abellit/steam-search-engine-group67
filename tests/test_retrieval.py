from src.retrieval import search

query = "multiplayer shooter"

results = search(query)

for i, r in enumerate(results):
    print(f"\nResult {i+1}")
    print("Score:", r["score"])
    print("Title:", r["doc"]["title"])