import argparse
import pandas as pd
from src.preprocessing import preprocess_dataset
from index import get_client, create_index, index_documents

def main():
    parser = argparse.ArgumentParser()   #parse cli argument
    parser.add_argument("--data", required=True, help="Path to dataset CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.data)   #loading dataset
    print(f"Loaded dataset: {len(df)} rows")

    docs = preprocess_dataset(df)  #preprocess
    print("Preprocessing complete")

    es = get_client()  #connect to elasticsearch

    create_index(es)

    index_documents(es, docs)


if __name__ == "__main__":
    main()