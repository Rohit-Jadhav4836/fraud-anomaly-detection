# Fraud & Anomaly Detection System

An unsupervised machine learning system that detects anomalous credit card transactions using Isolation Forest — without ever seeing fraud labels during training. Built on the real Kaggle/ULB Credit Card Fraud dataset.

🔗 **Live Demo:** 

## Dashboard Preview

<img width="1916" height="911" alt="ss-1" src="https://github.com/user-attachments/assets/adcd95e1-529d-4600-bcdb-d463c46afcd1" />



<img width="1915" height="867" alt="ss-2" src="https://github.com/user-attachments/assets/b1c24913-92cf-4359-97df-e462f60c3001" />




<img width="1911" height="910" alt="ss-3" src="https://github.com/user-attachments/assets/f4242ea0-d874-4dad-88e5-e8e5e22cd1f4" />





## Overview

Most portfolio fraud-detection projects use supervised classification with labeled fraud data — but in the real world, fraud patterns are often unknown or unlabeled when they first emerge. This project instead uses **unsupervised anomaly detection (Isolation Forest)**, which learns what "normal" transactions look like and flags outliers — then uses the real fraud labels only to evaluate performance afterward, not to train.

Built on the real **Kaggle Credit Card Fraud Detection** dataset (ULB) — 284,807 anonymized European transactions, 492 real frauds. This repo ships a representative sample (all 492 frauds + 40,000 sampled legitimate transactions) to keep the file size manageable while preserving every real fraud case.

Part of a Data Analytics / Data Science & AI-ML portfolio built during certification with IT Vedant.

## Features

- **Real, well-known dataset** (Kaggle/ULB Credit Card Fraud) — anonymized PCA features V1-V28
- **Unsupervised anomaly detection** using Isolation Forest — no fraud labels used at training time
- **Custom `SuspiciousTransactionError` exception** for real-time critical-anomaly flagging
- **10 interactive charts**: histogram, scatter, time series, donut, box plot, bracket analysis, correlation heatmap, grouped bar, gauge, treemap
- **CSV upload interface** with adjustable strict mode (raises exception on critical anomalies)
- **Neumorphic (soft UI) design** with layered gradient depth, light theme default

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Data | pandas, NumPy |
| ML | scikit-learn (Isolation Forest, StandardScaler) |
| Visualization | Plotly Express, Matplotlib, Seaborn |
| Web App | Streamlit |
| Styling | Custom CSS (neumorphism) |

## Project Structure

fraud-anomaly-detection/
├── assets/
│ └── style.css
├── data/
│ └── credit_card_transactions.csv
├── models/
│ └── anomaly_model.pkl
├── notebook/
│ └── 01_eda_and_modeling.ipynb
├── src/
│ ├── data_loader.py
│ ├── exceptions.py
│ ├── anomaly_model.py
│ └── detect.py
├── screenshots/
├── app.py
├── requirements.txt
└── README.md


## Model Approach

| Aspect | Detail |
|---|---|
| Algorithm | Isolation Forest (unsupervised) |
| Training data | V1–V28 (PCA features) + Amount — NO Class label used |
| Contamination parameter | 0.012 (matches the ~1.2% anomaly rate in the sampled data) |
| Evaluation | Fraud labels used post-hoc only, via classification report + ROC-AUC |
| Custom exception | `SuspiciousTransactionError` raised on critical anomaly scores for manual review workflows |

## About the Dataset Sample

The original Kaggle dataset (284,807 rows, ~98MB) was sampled down to keep the repo lightweight: **all 492 real fraud cases were retained**, combined with 40,000 randomly sampled legitimate transactions (40,492 rows total, 15MB). This shifts the fraud rate from the original ~0.17% to ~1.2% in this sample — noted here for transparency, since the two ratios shouldn't be confused when discussing results.

## Run Locally

```bash
git clone https://github.com/<your-username>/fraud-anomaly-detection.git
cd fraud-anomaly-detection
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run app.py
```

To retrain the model from scratch:
```bash
python src/anomaly_model.py
```

## What I Learned

- The difference between supervised and unsupervised approaches to fraud/anomaly detection, and when each is appropriate
- Isolation Forest's mechanism (isolating outliers via random partitioning) versus distance- or density-based anomaly methods
- Designing custom exceptions for domain-specific business logic (`SuspiciousTransactionError`) alongside a data pipeline
- Managing large real-world datasets for a GitHub-hosted portfolio (sampling strategy while preserving all rare-class examples)
