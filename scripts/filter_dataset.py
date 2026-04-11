import pandas as pd

# Downloading game file from Kaggle
df = pd.read_csv("data\cleaned_data\games_may2024_cleaned.csv")

print(f"Full dataset: {len(df)} games")
print(f"Columns: {df.columns.tolist()}")

# Saving a 5000 game sample
sample = df.sample(n=5000, random_state=42)
sample.to_csv('games_sample.csv', index=False)

print(f"Sample saved: {len(sample)} games")
