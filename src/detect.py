import pandas as pd
import numpy as np
import joblib
from src.exceptions import SuspiciousTransactionError

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Anomaly score threshold above which we raise SuspiciousTransactionError
# for manual-review-worthy cases (tunable, top ~0.1% most extreme scores)
CRITICAL_SCORE_THRESHOLD = 0.65


def load_anomaly_bundle(path: str = "models/anomaly_model.pkl"):
    return joblib.load(path)


def detect_anomalies(df: pd.DataFrame, bundle: dict, raise_on_critical: bool = False) -> pd.DataFrame:
    """
    Score transactions for anomalousness using the trained Isolation Forest.
    If raise_on_critical=True, raises SuspiciousTransactionError the moment
    a transaction's anomaly score crosses CRITICAL_SCORE_THRESHOLD —
    useful to demonstrate real-time flagging / exception handling.
    """
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_cols"]

    working = df.copy()
    X = working[feature_cols]
    X_scaled = scaler.transform(X)

    raw_pred = model.predict(X_scaled)
    anomaly_flag = (raw_pred == -1).astype(int)
    anomaly_score = -model.decision_function(X_scaled)

    # Normalize score to a readable 0-1ish range for display
    normalized_score = (anomaly_score - anomaly_score.min()) / (anomaly_score.max() - anomaly_score.min() + 1e-9)

    working["Anomaly_Score"] = normalized_score.round(4)
    working["Is_Anomaly"] = pd.Series(anomaly_flag, index=working.index).map({1: "Anomalous", 0: "Normal"})

    if raise_on_critical:
        critical_rows = working[working["Anomaly_Score"] >= CRITICAL_SCORE_THRESHOLD]
        if not critical_rows.empty:
            first = critical_rows.iloc[0]
            raise SuspiciousTransactionError(
                transaction_id=first.name,
                anomaly_score=first["Anomaly_Score"],
                amount=first["Amount"]
            )

    return working