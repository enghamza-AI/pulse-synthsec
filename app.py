
from __future__ import annotations

import streamlit as st
import plotly.express as px

from src.config_loader import load_config, get_data_mode
from src.pipeline import run_pipeline

st.set_page_config(
    page_title="Pulse — Customer Intelligence Dashboard",
    page_icon=":material/insights:",
    layout="wide",
)



@st.cache_data(show_spinner=False)
def cached_load_config() -> dict:
    """Config is static per deploy — safe to cache indefinitely."""
    return load_config()


@st.cache_data(show_spinner=False)
def cached_run_pipeline(data_mode: str) -> dict:
    """
    The one expensive call in the whole app. Cached on data_mode so
    switching between demo/local doesn't return a stale result, and so
    re-running with the SAME mode (e.g. a page rerun from an unrelated
    widget) doesn't recompute anything.
    """
    config = cached_load_config()
    output = run_pipeline(config, data_mode=data_mode)
  
    return output



with st.sidebar:
    st.markdown("### Pulse")
    st.caption("SynthSec customer intelligence demo")

    config = cached_load_config()
    default_mode = get_data_mode(config)

    data_mode = st.radio(
        "Data mode",
        options=["demo", "local"],
        index=0 if default_mode == "demo" else 1,
        help=(
            "demo: pre-generated 2,000-row sample, fast on free tier. "
            "local: full synthetic generation, for development only."
        ),
    )

    st.divider()
    st.caption(
        "All data on this page is synthetic — generated to demonstrate "
        "the methodology, never real client data."
    )




st.title("Pulse — which customers matter?")
st.markdown(
    "A live demo of SynthSec's customer intelligence methodology, built "
    "entirely on **synthetic** ecommerce data."
)

with st.expander("What is this, and what problem does it solve?", expanded=True):
    st.markdown(
        """
Most ecommerce tools try to predict "churn" the way a subscription
business would. That's the wrong question for physical products — a
customer doesn't "cancel" a t-shirt. The real question is: **when is
this specific customer likely to buy again, and is it worth reaching
out before then?**

Pulse answers that in two steps:

1. **Segmentation** (unsupervised) — groups customers into VIP,
   regular, at-risk, and one-time buyers based on real behavior, not
   demographics.
2. **Repurchase-window scoring** (supervised) — predicts, in days, when
   each customer is actually likely to buy again, so outreach timing
   isn't a generic 30-day guess.
        """
    )

st.divider()


if "pipeline_output" not in st.session_state:
    st.session_state.pipeline_output = None
if "last_data_mode" not in st.session_state:
    st.session_state.last_data_mode = None

run_col, status_col = st.columns([1, 3])
with run_col:
    run_clicked = st.button("Run analysis", type="primary", use_container_width=True)

if run_clicked:
    with st.status("Running Pulse pipeline...", expanded=True) as status:
        st.write("Loading data...")
        st.write("Cleaning and engineering features...")
        st.write("Segmenting customers (K-Means)...")
        st.write("Training repurchase-window model...")
        output = cached_run_pipeline(data_mode)
        st.session_state.pipeline_output = output
        st.session_state.last_data_mode = data_mode
        status.update(label="Analysis complete", state="complete", expanded=False)

output = st.session_state.pipeline_output

if output is None:
    st.info("Click **Run analysis** to generate the dashboard.")
    st.stop()

if st.session_state.last_data_mode != data_mode:
    st.warning(
        f"Showing results for '{st.session_state.last_data_mode}' mode — "
        f"click Run analysis again to refresh for '{data_mode}' mode."
    )

result = output["result"]



st.divider()
st.subheader("Dashboard")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Customers analyzed", f"{output['n_customers']:,}")
m2.metric("Segments found", result["segment_name"].nunique())
m3.metric("Model MAE (days)", output["model_mae"])
m4.metric("Silhouette score", output["silhouette_avg"])



segment_counts = result["segment_name"].value_counts().reset_index()
segment_counts.columns = ["segment", "count"]

fig_segments = px.bar(
    segment_counts,
    x="segment",
    y="count",
    color="segment",
    title="Customers per segment",
)
fig_segments.update_layout(showlegend=False, height=360)
st.plotly_chart(fig_segments, use_container_width=True)
del segment_counts, fig_segments  # no longer needed after rendering



st.subheader("Insights")

fig_scatter = px.scatter(
    result,
    x="recency_days",
    y="predicted_days_to_next_purchase",
    color="segment_name",
    hover_data=["customer_id", "monetary_total", "frequency"],
    title="Recency vs. predicted repurchase window, by segment",
)
fig_scatter.update_layout(height=420)
st.plotly_chart(fig_scatter, use_container_width=True)
del fig_scatter

urgency_counts = result["urgency"].value_counts().reindex(
    ["high", "medium", "low"]
).fillna(0).reset_index()
urgency_counts.columns = ["urgency", "count"]
fig_urgency = px.bar(
    urgency_counts,
    x="urgency",
    y="count",
    color="urgency",
    color_discrete_map={"high": "#D85A30", "medium": "#EF9F27", "low": "#5DCAA5"},
    title="Customers by recommended-action urgency",
)
fig_urgency.update_layout(showlegend=False, height=320)
st.plotly_chart(fig_urgency, use_container_width=True)
del urgency_counts, fig_urgency



st.subheader("Recommended actions")
st.caption("Sorted by urgency — start at the top.")

urgency_order = {"high": 0, "medium": 1, "low": 2}
display_df = result.copy()
display_df["_sort"] = display_df["urgency"].map(urgency_order)
display_df = display_df.sort_values(["_sort", "recency_days"], ascending=[True, False])

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
del display_df  

st.divider()
st.caption(
    "Pulse is a demo built on synthetic data. A client engagement runs "
    "the identical pipeline against real Shopify/CRM exports — see "
    "about_the_project.md for how that swap works."
)
