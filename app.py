import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.detect import load_anomaly_bundle, detect_anomalies
from src.exceptions import SuspiciousTransactionError

st.set_page_config(page_title="Fraud & Anomaly Detection", layout="wide")

LIGHT_THEME_CSS = """
:root {
  --card-bg: #ffffff;
  --card-border: rgba(15, 23, 42, 0.07);
  --shadow-color: rgba(15, 23, 42, 0.10);
  --shadow-color-soft: rgba(15, 23, 42, 0.06);
  --dropzone-bg: #f8fafc;
  --sidebar-bg: #ffffff;
  --accent-green: #34d399;
  --accent-dark: #1f2937;
  --accent-blue: #60a5fa;
  --accent-amber: #fbbf24;
  --text-color: #1f2937;
}
[data-testid="stAppViewContainer"] {
  background: linear-gradient(135deg, #eafff2, #e6f7ff, #eafff5);
  background-size: 300% 300%;
  animation: bgDrift 30s ease infinite;
}
[data-testid="stAppViewContainer"] *:not([data-testid="stDateInput"] *) { color: var(--text-color); }
[data-testid="stSidebar"] *:not([data-testid="stDateInput"] *) { color: var(--text-color) !important; }
[data-testid="stHeader"] { background: rgba(234, 255, 242, 0.6) !important; }
"""

DARK_THEME_CSS = """
:root {
  --card-bg: #1a1d24;
  --card-border: rgba(255, 255, 255, 0.08);
  --shadow-color: rgba(0, 0, 0, 0.35);
  --shadow-color-soft: rgba(0, 0, 0, 0.5);
  --dropzone-bg: #14161b;
  --sidebar-bg: #14161b;
  --accent-green: #34d399;
  --accent-dark: #f3f4f6;
  --accent-blue: #60a5fa;
  --accent-amber: #fbbf24;
  --text-color: #e5e7eb;
}
[data-testid="stAppViewContainer"] {
  background: #0f1115;
}
[data-testid="stAppViewContainer"] * { color: var(--text-color); }
[data-testid="stHeader"] { background: #0f1115 !important; }
[data-testid="stFileUploader"] section button { background: var(--accent-green) !important; color: #0f1115 !important; }
.stButton button, .stDownloadButton button { background: var(--accent-green) !important; color: #0f1115 !important; }
"""

LOGO_SVG = """
<div class="sidebar-logo-wrap">
  <div class="sidebar-logo">
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 2 L20 6 V12 C20 17 16.5 20.5 12 22 C7.5 20.5 4 17 4 12 V6 Z" stroke="#34d399" stroke-width="1.6" fill="none"/>
      <path d="M9 12 L11 14 L15.5 9.5" stroke="#34d399" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    <span>SENTRYNET</span>
    <div class="led-dot"></div>
  </div>
</div>
"""

def load_css(theme_mode: str, path: str = "assets/style.css"):
    with open(path) as f:
        base_css = f.read()
    theme_css = LIGHT_THEME_CSS if theme_mode == "light" else DARK_THEME_CSS
    st.markdown(f"<style>{base_css}\n{theme_css}</style>", unsafe_allow_html=True)

def centered_title(text: str):
    st.markdown(f"<h3 style='text-align:center;'>{text}</h3>", unsafe_allow_html=True)

CLEAN_PALETTE = ["#34d399", "#1f2937", "#60a5fa", "#fbbf24", "#f87171", "#a78bfa"]

def apply_theme(fig, theme_mode: str):
    text_color = "#1f2937" if theme_mode == "light" else "#e5e7eb"
    grid_color = "rgba(15,23,42,0.06)" if theme_mode == "light" else "rgba(255,255,255,0.06)"
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=text_color),
        colorway=CLEAN_PALETTE,
        margin=dict(l=20, r=20, t=10, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=text_color)),
        height=380,
    )
    fig.update_xaxes(showgrid=True, gridcolor=grid_color, zeroline=False,
                      tickfont=dict(color=text_color), title_font=dict(color=text_color))
    fig.update_yaxes(showgrid=True, gridcolor=grid_color, zeroline=False,
                      tickfont=dict(color=text_color), title_font=dict(color=text_color))
    return fig

def safe_chart(build_fn, theme_mode: str):
    try:
        fig = build_fn()
        st.plotly_chart(apply_theme(fig, theme_mode), use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render this chart: {e}")

# ---------------- SIDEBAR ----------------
st.sidebar.markdown(LOGO_SVG, unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-subtitle">TRANSACTION SECURITY MONITOR</div>', unsafe_allow_html=True)

theme_choice = st.sidebar.radio("Appearance", ["Light", "Dark"], horizontal=True, label_visibility="collapsed")
theme_mode = "light" if theme_choice == "Light" else "dark"

load_css(theme_mode)

st.title("Fraud & Anomaly Detection")
st.caption("Unsupervised anomaly detection on credit card transactions using Isolation Forest, no fraud labels used at training time.")

bundle = load_anomaly_bundle()

st.sidebar.markdown("### Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload transaction CSV", type=["csv"])
strict_mode = st.sidebar.checkbox("Raise exception on critical anomalies", value=False)

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)

    try:
        results = detect_anomalies(raw_df, bundle, raise_on_critical=strict_mode)
    except SuspiciousTransactionError as e:
        st.error(f"SuspiciousTransactionError raised: {e}")
        st.stop()

    anomaly_count = (results["Is_Anomaly"] == "Anomalous").sum()
    total_flagged_amount = results.loc[results["Is_Anomaly"] == "Anomalous", "Amount"].sum()
    avg_score = results["Anomaly_Score"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transactions Analyzed", len(results))
    col2.metric("Anomalies Flagged", anomaly_count)
    col3.metric("Flagged Amount", f"₹{total_flagged_amount:,.0f}")
    col4.metric("Avg Anomaly Score", f"{avg_score:.3f}")

    st.divider()

    risk_colors = {"Anomalous": "#f87171", "Normal": "#34d399"}

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            centered_title("Anomaly Score Distribution")
            safe_chart(lambda: px.histogram(results, x="Anomaly_Score", nbins=40, color="Is_Anomaly",
                                             color_discrete_map=risk_colors), theme_mode)
    with c2:
        with st.container(border=True):
            centered_title("Transaction Amount vs Anomaly Score")
            safe_chart(lambda: px.scatter(results, x="Amount", y="Anomaly_Score", color="Is_Anomaly",
                                           opacity=0.6, color_discrete_map=risk_colors), theme_mode)

    c3, c4 = st.columns(2)
    with c3:
        with st.container(border=True):
            centered_title("Anomalies Over Time")
            def _time_series():
                ts = results.copy()
                ts["Time_Bucket"] = (ts["Time"] // 3600).astype(int)
                grouped = ts.groupby(["Time_Bucket", "Is_Anomaly"]).size().reset_index(name="count")
                return px.line(grouped, x="Time_Bucket", y="count", color="Is_Anomaly", markers=True,
                                color_discrete_map=risk_colors)
            safe_chart(_time_series, theme_mode)
    with c4:
        with st.container(border=True):
            centered_title("Anomaly Share")
            def _donut():
                counts = results["Is_Anomaly"].value_counts().reset_index()
                counts.columns = ["Is_Anomaly", "count"]
                return px.pie(counts, names="Is_Anomaly", values="count", hole=0.55,
                              color="Is_Anomaly", color_discrete_map=risk_colors)
            safe_chart(_donut, theme_mode)

    c5, c6 = st.columns(2)
    with c5:
        with st.container(border=True):
            centered_title("Amount Distribution by Classification")
            safe_chart(lambda: px.box(results, x="Is_Anomaly", y="Amount", color="Is_Anomaly",
                                       color_discrete_map=risk_colors), theme_mode)
    with c6:
        with st.container(border=True):
            centered_title("Anomaly Rate by Amount Bracket")
            def _bracket():
                b = results.copy()
                b["Amount_Bracket"] = pd.cut(b["Amount"], bins=[0, 50, 200, 500, 1000, 5000, 1e9],
                                              labels=["0-50", "50-200", "200-500", "500-1K", "1K-5K", "5K+"])
                rate = b.groupby("Amount_Bracket", observed=True)["Is_Anomaly"].apply(
                    lambda x: (x == "Anomalous").mean() * 100
                ).reset_index(name="anomaly_rate_pct")
                return px.bar(rate, x="Amount_Bracket", y="anomaly_rate_pct",
                               color="anomaly_rate_pct", color_continuous_scale=["#34d399", "#f87171"])
            safe_chart(_bracket, theme_mode)

    c7, c8 = st.columns(2)
    with c7:
        with st.container(border=True):
            centered_title("Feature Correlation (V1-V10)")
            def _heatmap():
                subset = results[[f"V{i}" for i in range(1, 11)]]
                return px.imshow(subset.corr(), color_continuous_scale=["#34d399", "#ffffff", "#f87171"],
                                  zmin=-1, zmax=1, aspect="auto")
            safe_chart(_heatmap, theme_mode)
    with c8:
        with st.container(border=True):
            centered_title("Avg Feature Value: Anomalous vs Normal")
            def _feature_compare():
                feats = [f"V{i}" for i in range(1, 6)]
                means = results.groupby("Is_Anomaly")[feats].mean().reset_index().melt(
                    id_vars="Is_Anomaly", var_name="Feature", value_name="Mean_Value")
                return px.bar(means, x="Feature", y="Mean_Value", color="Is_Anomaly",
                              barmode="group", color_discrete_map=risk_colors)
            safe_chart(_feature_compare, theme_mode)

    c9, c10 = st.columns(2)
    with c9:
        with st.container(border=True):
            centered_title("Overall Anomaly Rate")
            def _gauge():
                rate = (results["Is_Anomaly"] == "Anomalous").mean() * 100
                return go.Figure(go.Indicator(
                    mode="gauge+number", value=rate, number={"suffix": "%"},
                    gauge={"axis": {"range": [0, 10]}, "bar": {"color": "#f87171"},
                           "steps": [{"range": [0, 2], "color": "#d1fae5"},
                                     {"range": [2, 5], "color": "#fef3c7"},
                                     {"range": [5, 10], "color": "#fee2e2"}]}
                ))
            safe_chart(_gauge, theme_mode)
    with c10:
        with st.container(border=True):
            centered_title("Amount Bracket Composition")
            def _treemap():
                b = results.copy()
                b["Amount_Bracket"] = pd.cut(b["Amount"], bins=[0, 50, 200, 500, 1000, 5000, 1e9],
                                              labels=["0-50", "50-200", "200-500", "500-1K", "1K-5K", "5K+"])
                agg = b.groupby(["Amount_Bracket", "Is_Anomaly"], observed=True).size().reset_index(name="count")
                return px.treemap(agg, path=["Amount_Bracket", "Is_Anomaly"], values="count",
                                   color="Is_Anomaly", color_discrete_map=risk_colors)
            safe_chart(_treemap, theme_mode)

    with st.container(border=True):
        centered_title("Flagged Anomalous Transactions")
        anomalies_df = results[results["Is_Anomaly"] == "Anomalous"].sort_values("Anomaly_Score", ascending=False)
        st.dataframe(anomalies_df[["Time", "Amount", "Anomaly_Score", "Is_Anomaly"]], use_container_width=True)

else:
    st.info("Upload a transaction CSV (same schema as the credit card dataset: Time, V1-V28, Amount) to run detection.")