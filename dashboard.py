import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Hybrid Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ============================================================
# LOAD PUBLIC DASHBOARD DATA
# ============================================================

@st.cache_data
def load_metrics():

    with open("dashboard_metrics.json", "r") as f:
        return json.load(f)


@st.cache_data
def load_sample():

    df = pd.read_csv("dashboard_data.csv")

    df["trans_date_trans_time"] = pd.to_datetime(
        df["trans_date_trans_time"]
    )

    return df


metrics = load_metrics()
df = load_sample()

# ============================================================
# HEADER
# ============================================================

st.title("💳 Hybrid Credit Card Fraud Detection")

st.markdown(
    """
### Two-Stage ML + Customer Behavioral Verification

**Stage 1:** Transaction-level Random Forest  
**Stage 2:** Customer behavioral Random Forest  
**Hybrid:** 70% ML + 30% Behavioral Probability
"""
)

st.divider()

# ============================================================
# EXACT KPIs
# ============================================================

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Transactions",
    f"{metrics['total_transactions']:,}"
)

c2.metric(
    "Fraud Cases",
    f"{metrics['fraud_transactions']:,}"
)

c3.metric(
    "Routing Recall",
    f"{metrics['routing_recall']:.2%}"
)

c4.metric(
    "Block Precision",
    f"{metrics['block_precision']:.2%}"
)

c5.metric(
    "Fraud Blocked",
    f"{metrics['fraud_blocked']:,}"
)

# ============================================================
# DECISION ENGINE
# ============================================================

st.divider()

st.subheader("Decision Engine")

a, b, c = st.columns(3)

a.metric(
    "🟢 APPROVE",
    f"{metrics['approve']:,}"
)

b.metric(
    "🟡 VERIFY",
    f"{metrics['verify']:,}"
)

c.metric(
    "🔴 BLOCK",
    f"{metrics['block']:,}"
)

st.caption(
    "APPROVE < 0.10  |  "
    "VERIFY 0.10–<0.90  |  "
    "BLOCK ≥ 0.90"
)

# ============================================================
# CHARTS
# ============================================================

left, right = st.columns(2)

# ------------------------------------------------------------
# DECISION DISTRIBUTION
# ------------------------------------------------------------

with left:

    st.subheader("Transaction Routing")

    decision_df = pd.DataFrame({
        "Decision": [
            "APPROVE",
            "VERIFY",
            "BLOCK"
        ],
        "Transactions": [
            metrics["approve"],
            metrics["verify"],
            metrics["block"]
        ]
    })

    fig = px.bar(
        decision_df,
        x="Decision",
        y="Transactions",
        text="Transactions"
    )

    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="outside"
    )

    fig.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ------------------------------------------------------------
# EXACT HYBRID HISTOGRAM
# ------------------------------------------------------------

with right:

    st.subheader("Hybrid Risk Distribution")

    hist = metrics["histogram"]

    edges = np.array(hist["edges"])
    counts = np.array(hist["counts"])

    centers = (
        edges[:-1] + edges[1:]
    ) / 2

    widths = np.diff(edges)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=centers,
            y=counts,
            width=widths,
            name="Transactions"
        )
    )

    fig.add_vline(
        x=0.10,
        line_dash="dash",
        annotation_text="VERIFY"
    )

    fig.add_vline(
        x=0.90,
        line_dash="dash",
        annotation_text="BLOCK"
    )

    fig.update_layout(
        xaxis_title="Hybrid Probability",
        yaxis_title="Transactions",
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ============================================================
# FRAUD RATE BY DECISION
# ============================================================

st.divider()

st.subheader("Fraud Rate by Decision")

stats = metrics["decision_stats"]

risk = pd.DataFrame({
    "Decision": [
        "APPROVE",
        "VERIFY",
        "BLOCK"
    ],
    "Transactions": [
        stats["APPROVE"]["transactions"],
        stats["VERIFY"]["transactions"],
        stats["BLOCK"]["transactions"]
    ],
    "Fraud Cases": [
        stats["APPROVE"]["fraud_cases"],
        stats["VERIFY"]["fraud_cases"],
        stats["BLOCK"]["fraud_cases"]
    ],
    "Fraud Rate (%)": [
        stats["APPROVE"]["fraud_rate"],
        stats["VERIFY"]["fraud_rate"],
        stats["BLOCK"]["fraud_rate"]
    ]
})

st.dataframe(
    risk,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Fraud Rate (%)": st.column_config.NumberColumn(
            format="%.4f%%"
        )
    }
)

# ============================================================
# MODEL COMPARISON
# ============================================================

st.divider()

st.subheader("Stage-1 vs Behavioral Risk")

fig = px.scatter(
    df,
    x="fraud_probability",
    y="behavior_probability",
    color="decision",
    hover_data=[
        "amt",
        "merchant",
        "category",
        "hybrid_probability"
    ],
    opacity=0.55
)

fig.update_layout(
    xaxis_title="Stage-1 Fraud Probability",
    yaxis_title="Behavioral Probability"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ============================================================
# TRANSACTION EXPLORER
# ============================================================

st.divider()

st.subheader("🔎 Transaction Explorer")

st.caption(
    "The explorer uses a representative public sample of "
    "10,000 transactions. Final KPIs above are calculated "
    "from all 555,719 evaluated transactions."
)

col1, col2 = st.columns(2)

with col1:

    decisions = st.multiselect(
        "Decision",
        ["APPROVE", "VERIFY", "BLOCK"],
        default=["VERIFY", "BLOCK"]
    )

with col2:

    score_range = st.slider(
        "Hybrid Score",
        0.0,
        1.0,
        (0.0, 1.0),
        0.01
    )

filtered = df[
    df["decision"].isin(decisions) &
    (df["hybrid_probability"] >= score_range[0]) &
    (df["hybrid_probability"] <= score_range[1])
]

display_columns = [
    "trans_date_trans_time",
    "amt",
    "merchant",
    "category",
    "fraud_probability",
    "behavior_probability",
    "hybrid_probability",
    "decision"
]

st.write(
    f"{len(filtered):,} sample transactions match the filters"
)

st.dataframe(
    filtered[display_columns],
    use_container_width=True,
    hide_index=True
)

# ============================================================
# RISK SCORE SIMULATOR
# ============================================================

st.divider()

st.subheader("🛡️ Risk Score Simulator")

st.write(
    "Simulate the final decision using Stage-1 and "
    "behavioral probabilities."
)

x, y = st.columns(2)

with x:

    stage1 = st.slider(
        "Stage-1 Fraud Probability",
        0.0,
        1.0,
        0.50,
        0.01
    )

with y:

    behavior = st.slider(
        "Behavioral Probability",
        0.0,
        1.0,
        0.20,
        0.01
    )

hybrid = (
    0.70 * stage1 +
    0.30 * behavior
)

if hybrid < 0.10:

    decision = "APPROVE"

    message = (
        "Low combined risk. Transaction can proceed."
    )

elif hybrid < 0.90:

    decision = "VERIFY"

    message = (
        "Suspicious transaction. Additional customer "
        "verification is recommended."
    )

else:

    decision = "BLOCK"

    message = (
        "Extremely high combined risk. Transaction "
        "should be blocked."
    )

r1, r2 = st.columns(2)

r1.metric(
    "Hybrid Score",
    f"{hybrid:.3f}"
)

r2.metric(
    "Recommended Decision",
    decision
)

st.info(message)

st.divider()

st.caption(
    "Synthetic dataset • Random Forest • Behavioral "
    "Verification • Hybrid Fraud Detection"
)
