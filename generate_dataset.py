import os
import numpy as np
import pandas as pd

np.random.seed(42)

OUTPUT_DIR = "data/original"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "transactions.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

n = 20000

transaction_id = [f"TXN{100000 + i}" for i in range(n)]

amount = np.round(np.random.lognormal(mean=7.0, sigma=1.0, size=n), 2)
amount = np.clip(amount, 50, 200000)

hour = np.random.randint(0, 24, n)

device_type = np.random.choice(
    ["Mobile", "Desktop", "Tablet"],
    size=n,
    p=[0.65, 0.25, 0.10]
)

location = np.random.choice(
    ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Chennai", "Pune", "Kolkata"],
    size=n
)

account_age_days = np.random.randint(30, 3000, n)

transaction_frequency = np.random.randint(1, 30, n)

failed_attempts = np.random.poisson(0.5, n)
failed_attempts = np.clip(failed_attempts, 0, 8)

distance_from_last_transaction = np.round(
    np.random.exponential(scale=20, size=n),
    2
)

previous_transaction_amount = np.round(
    np.random.lognormal(mean=6.5, sigma=0.8, size=n),
    2
)

previous_transaction_amount = np.clip(
    previous_transaction_amount,
    50,
    150000
)

new_device = np.random.choice(
    [0, 1],
    size=n,
    p=[0.85, 0.15]
)

location_changed = np.random.choice(
    [0, 1],
    size=n,
    p=[0.80, 0.20]
)

night_transaction = ((hour <= 5) | (hour >= 23)).astype(int)

amount_ratio = (
    amount / (previous_transaction_amount + 1)
)

fraud_probability = (
    0.02
    + 0.12 * (amount > 50000)
    + 0.15 * (amount > 100000)
    + 0.12 * night_transaction
    + 0.15 * new_device
    + 0.18 * location_changed
    + 0.08 * (failed_attempts >= 2)
    + 0.12 * (distance_from_last_transaction > 100)
    + 0.10 * (amount_ratio > 4)
)

fraud_probability = np.clip(fraud_probability, 0, 0.90)

is_fraud = np.random.binomial(
    1,
    fraud_probability
)

df = pd.DataFrame({
    "transaction_id": transaction_id,
    "amount": amount,
    "hour": hour,
    "device_type": device_type,
    "location": location,
    "account_age_days": account_age_days,
    "transaction_frequency": transaction_frequency,
    "failed_attempts": failed_attempts,
    "distance_from_last_transaction": distance_from_last_transaction,
    "previous_transaction_amount": previous_transaction_amount,
    "new_device": new_device,
    "location_changed": location_changed,
    "night_transaction": night_transaction,
    "amount_ratio": amount_ratio,
    "is_fraud": is_fraud
})

df.to_csv(OUTPUT_FILE, index=False)

print("Dataset generated successfully.")
print(f"File: {OUTPUT_FILE}")
print(f"Total transactions: {len(df)}")
print(f"Fraud transactions: {df['is_fraud'].sum()}")
print(f"Normal transactions: {len(df) - df['is_fraud'].sum()}")
print("\nFirst 5 transactions:")
print(df.head())