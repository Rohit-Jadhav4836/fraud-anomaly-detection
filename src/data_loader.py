import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def load_transactions(filepath: str = "data/credit_card_transactions.csv") -> pd.DataFrame:
    """Load the credit card transactions dataset (PCA-anonymized features V1-V28, Time, Amount, Class)."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df