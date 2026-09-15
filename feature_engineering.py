import pandas as pd


def create_features(df):
    data = df.copy()

    data["amount_log"] = (
        data["amount"].apply(lambda x: __import__("math").log1p(x))
    )

    data["high_amount"] = (
        data["amount"] > 50000
    ).astype(int)

    data["high_frequency"] = (
        data["transaction_frequency"] > 15
    ).astype(int)

    data["multiple_failed_attempts"] = (
        data["failed_attempts"] >= 2
    ).astype(int)

    data["large_distance"] = (
        data["distance_from_last_transaction"] > 100
    ).astype(int)

    data["unusual_amount"] = (
        data["amount_ratio"] > 4
    ).astype(int)

    return data