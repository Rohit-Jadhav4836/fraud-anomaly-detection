class SuspiciousTransactionError(Exception):
    """Raised when a transaction is flagged as highly suspicious
    (extreme anomaly score) and requires manual review."""

    def __init__(self, transaction_id, anomaly_score, amount):
        self.transaction_id = transaction_id
        self.anomaly_score = anomaly_score
        self.amount = amount
        super().__init__(
            f"Suspicious transaction detected [ID={transaction_id}] "
            f"Amount=₹{amount:,.2f} AnomalyScore={anomaly_score:.3f}"
        )