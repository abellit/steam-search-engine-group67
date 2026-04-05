from src.retrieval import search
import ir_measures
from ir_measures import MAP, MRR, nDCG, P
from collections import namedtuple


def load_queries(path):
    queries = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("```"):
                continue

            queries.append(line)

    return queries


Qrel = namedtuple("Qrel", ["query_id", "doc_id", "relevance"])


def load_qrels(path):
    qrels = []

    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue

            qid, _, doc_id, rel = line.split()

            qrels.append(
                Qrel(
                    query_id=qid,
                    doc_id=doc_id,
                    relevance=int(rel)
                )
            )

    return qrels


ScoredDoc = namedtuple("ScoredDoc", ["query_id", "doc_id", "score"])


def run_system(queries):
    run = []

    for i, q in enumerate(queries):
        qid = str(i + 1)

        results = search(q, top_k=10)

        for rank, r in enumerate(results):
            run.append(
                ScoredDoc(
                    query_id=qid,
                    doc_id=r["doc_id"], 
                    score=r["score"]
                )
            )
    print(f"\nQuery {qid}: {q}")
    print("Top results:", results[:3])
    return run


def evaluate(qrels, run):
    measures = [MAP, MRR, nDCG@10, P@10]

    results = ir_measures.calc_aggregate(measures, qrels, run)

    print("\nEvaluation Results:")
    for m in measures:
        print(f"{m}: {results[m]}")