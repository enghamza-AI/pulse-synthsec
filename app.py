from __future__ import annotations

import pickle
from pathlib import Path

import streamlit as st
import plotly.express as px


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Pulse — Customer Intelligence Dashboard",
    page_icon=":material/insights:",
    layout="wide",
)


# ============================================================
# LOAD PRECOMPUTED RESULTS
# ============================================================

@st.cache_resource
def load_results():
    """
    Load the precomputed Pulse results.

    The expensive ML pipeline is NOT run by Streamlit.
    generate_demo.py already ran:
        - data cleaning
        - feature engineering
        - K-Means segmentation
        - repurchase model training
        - predictions
        - recommendations

    Streamlit only loads the resulting pickle.
    """

    project_root = Path(__file__).resolve().parent
    results_path = project_root / "data" / "demo" / "pulse_results.pkl"

    if not results_path.exists():
        raise FileNotFoundError(
            f"Pulse results file not found: {results_path}\n\n"
            "Run:\n"
            "python scripts/generate_demo.py\n"
            "locally first, then push the generated .pkl file to GitHub."
        )

    with open(results_path, "rb") as f:
        return pickle.load(f)


# ============================================================
# LOAD DATA
# ============================================================

output = load_results()
result = output["result"]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### Pulse")
    st.caption("SynthSec customer intelligence demo")

    st.divider()

    st.caption(
        "All data on this page is synthetic — generated to demonstrate "
        "the methodology, never real client data."
    )


# ============================================================
# HEADER
# ============================================================

st.title("Pulse — which customers matter?")

st.markdown(
    "A live demo of SynthSec's customer intelligence methodology, built "
    "entirely on **synthetic ecommerce data**."
)


# ============================================================
# EXPLANATION
# ============================================================

with st.expander(
    "What is this, and what problem does it solve?",
    expanded=True,
):
    st.markdown(
        """
Most ecommerce tools try to predict "churn" the way a subscription
business would. That's the wrong question for physical products — a
customer doesn't "cancel" a t-shirt.

The real question is:

**When is this specific customer likely to buy again, and is it worth
reaching out before then?**

Pulse answers that in two steps:

1. **Segmentation (unsupervised)** — groups customers into VIP,
   regular, at-risk, and one-time buyers based on real behavior,
   not demographics.

2. **Repurchase-window scoring (supervised)** — predicts, in days,
   when each customer is likely to buy again, so outreach timing
   isn't based on a generic 30-day guess.
        """
    )


st.divider()


# ============================================================
# DASHBOARD
# ============================================================

st.subheader("Dashboard")


m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Customers analyzed",
    f"{output['n_customers']:,}",
)

m2.metric(
    "Segments found",
    result["segment_name"].nunique(),
)

m3.metric(
    "Model MAE (days)",
    output["model_mae"],
)

m4.metric(
    "Silhouette score",
    output["silhouette_avg"],
)


# ============================================================
# SEGMENT DISTRIBUTION
# ============================================================

segment_counts = (
    result["segment_name"]
    .value_counts()
    .reset_index()
)

segment_counts.columns = ["segment", "count"]

fig_segments = px.bar(
    segment_counts,
    x="segment",
    y="count",
    color="segment",
    title="Customers per segment",
)

fig_segments.update_layout(
    showlegend=False,
    height=360,
)

st.plotly_chart(
    fig_segments,
    use_container_width=True,
)


# ============================================================
# INSIGHTS
# ============================================================

st.subheader("Insights")


fig_scatter = px.scatter(
    result,
    x="recency_days",
    y="predicted_days_to_next_purchase",
    color="segment_name",
    hover_data=[
        "customer_id",
        "monetary_total",
        "frequency",
    ],
    title=(
        "Recency vs. predicted repurchase window, "
        "by segment"
    ),
)

fig_scatter.update_layout(height=420)

st.plotly_chart(
    fig_scatter,
    use_container_width=True,
)


# ============================================================
# URGENCY
# ============================================================

urgency_counts = (
    result["urgency"]
    .value_counts()
    .reindex(["high", "medium", "low"])
    .fillna(0)
    .reset_index()
)

urgency_counts.columns = ["urgency", "count"]

fig_urgency = px.bar(
    urgency_counts,
    x="urgency",
    y="count",
    color="urgency",
    color_discrete_map={
        "high": "#D85A30",
        "medium": "#EF9F27",
        "low": "#5DCAA5",
    },
    title="Customers by recommended-action urgency",
)

fig_urgency.update_layout(
    showlegend=False,
    height=320,
)

st.plotly_chart(
    fig_urgency,
    use_container_width=True,
)


# ============================================================
# RECOMMENDED ACTIONS
# ============================================================

st.subheader("Recommended actions")

st.caption("Sorted by urgency — start at the top.")


urgency_order = {
    "high": 0,
    "medium": 1,
    "low": 2,
}

display_df = result.copy()

display_df["_sort"] = display_df["urgency"].map(
    urgency_order
)

display_df = display_df.sort_values(
    ["_sort", "recency_days"],
    ascending=[True, False],
)


st.dataframe(
    display_df[
        [
            "customer_id",
            "segment_name",
            "urgency",
            "predicted_days_to_next_purchase",
            "recommended_action",
        ]
    ].head(50),
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Pulse is a demo built on synthetic data. A client engagement runs "
    "the identical pipeline against real Shopify/CRM exports — see "
    "about_the_project.md for how that swap works."
)
