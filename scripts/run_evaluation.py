from src.evaluation import load_queries, load_qrels, run_system, evaluate

QUERIES_PATH = "queries/test_queries.txt"
QRELS_PATH = "queries/qrels.txt"

def main():
    queries = load_queries(QUERIES_PATH)
    qrels = load_qrels(QRELS_PATH)

    run = run_system(queries)

    evaluate(qrels, run)


if __name__ == "__main__":
    main()