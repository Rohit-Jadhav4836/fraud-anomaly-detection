import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]


def train_isolation_forest(df: pd.DataFrame, contamination: float = 0.012):
    """
    Train an unsupervised Isolation Forest to flag anomalous transactions.
    contamination = expected proportion of anomalies (~1.2% here, matching our sampled data).

    Note: we do NOT use the 'Class' label to train — this is unsupervised.
    We only use it afterward to EVALUATE how well the model's anomaly
    flags line up with real fraud labels.
    """
    X = df[FEATURE_COLS].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)

    # -1 = anomaly, 1 = normal -> convert to 1 = anomaly, 0 = normal
    raw_pred = model.predict(X_scaled)
    anomaly_flag = (raw_pred == -1).astype(int)

    # decision_function: higher = more normal. Flip sign so higher = more anomalous.
    anomaly_score = -model.decision_function(X_scaled)

    if "Class" in df.columns:
        print(classification_report(df["Class"], anomaly_flag, target_names=["Normal", "Fraud"]))
        print(f"ROC-AUC (using anomaly score vs true label): {roc_auc_score(df['Class'], anomaly_score):.3f}")

    joblib.dump({
        "model": model,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
    }, "models/anomaly_model.pkl")
    print("Model saved to models/anomaly_model.pkl")

    return model, scaler, anomaly_flag, anomaly_score


if __name__ == "__main__":
    from data_loader import load_transactions
    df = load_transactions.__wrapped__("data/credit_card_transactions.csv")
    train_isolation_forest(df)