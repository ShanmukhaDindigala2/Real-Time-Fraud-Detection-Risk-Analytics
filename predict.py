import joblib
import pandas as pd

from src.features.feature_engineering import create_features


MODEL_FILE = "ml_assets/fraud_model.pkl"
PREPROCESSOR_FILE = "ml_assets/scaler.pkl"


model = joblib.load(MODEL_FILE)
preprocessor = joblib.load(PREPROCESSOR_FILE)


def predict_transaction(transaction):
    df = pd.DataFrame([transaction])

    df = create_features(df)

    df = df.drop(
        columns=["transaction_id"],
        errors="ignore"
    )

    processed_data = preprocessor.transform(df)

    probability = model.predict_proba(
        processed_data
    )[0][1]

    risk_score = round(probability * 100, 2)

    if risk_score < 30:
        risk_level = "LOW"
    elif risk_score < 70:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    prediction = (
        "FRAUD RISK"
        if risk_score >= 70
        else "NORMAL"
    )

    return {
        "fraud_probability": round(
            probability,
            4
        ),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "prediction": prediction
    }