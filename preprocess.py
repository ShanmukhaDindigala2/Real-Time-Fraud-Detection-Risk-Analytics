import os
import pandas as pd

INPUT_FILE = "data/original/transactions.csv"
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "processed_transactions.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

print("Original dataset shape:", df.shape)

df = df.drop_duplicates()

df = df.dropna()

df["amount"] = df["amount"].clip(lower=0)

df["previous_transaction_amount"] = (
    df["previous_transaction_amount"].clip(lower=0)
)

df["amount_ratio"] = (
    df["amount"] /
    (df["previous_transaction_amount"] + 1)
)

df.to_csv(OUTPUT_FILE, index=False)

print("Preprocessing completed successfully.")
print("Processed dataset shape:", df.shape)
print(f"Saved to: {OUTPUT_FILE}")