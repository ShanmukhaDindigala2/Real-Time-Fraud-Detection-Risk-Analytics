import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

from src.features.feature_engineering import create_features


DATA_FILE = "data/processed/processed_transactions.csv"
MODEL_DIR = "ml_assets"

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_FILE)

df = create_features(df)

X = df.drop(
    columns=["transaction_id", "is_fraud"]
)

y = df["is_fraud"]

categorical_features = [
    "device_type",
    "location"
]

numerical_features = [
    column
    for column in X.columns
    if column not in categorical_features
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numerical_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(
    X_train_processed,
    y_train
)

predictions = model.predict(X_test_processed)

probabilities = model.predict_proba(
    X_test_processed
)[:, 1]

print("\nModel Evaluation")
print("================")

print(
    classification_report(
        y_test,
        predictions
    )
)

print(
    "ROC-AUC:",
    round(
        roc_auc_score(
            y_test,
            probabilities
        ),
        4
    )
)

joblib.dump(
    model,
    "ml_assets/fraud_model.pkl"
)

joblib.dump(
    preprocessor,
    "ml_assets/scaler.pkl"
)

feature_names = (
    preprocessor.get_feature_names_out().tolist()
)

joblib.dump(
    feature_names,
    "ml_assets/feature_columns.pkl"
)

print("\nModel saved successfully.")
print("Location: ml_assets/")