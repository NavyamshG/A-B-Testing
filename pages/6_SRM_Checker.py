import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from utils.theme import inject_base_css, page_header, description_box
from utils.stats_engine import srm_chi_square

st.set_page_config(page_title="SRM Checker", layout="wide")
inject_base_css()

page_header("Sample Ratio Mismatch (SRM) Checker", "The silent experiment killer.", "🚦")
description_box(
    "Run a chi-square goodness-of-fit test to check whether the observed split between Control "
    "and Variant matches the intended randomization ratio.",
    "If your assignment logic says 50/50 but you actually observe 51,000 vs 49,200 users, that "
    "imbalance can be innocent noise — or it can mean your randomization, logging, or bot-filtering "
    "is broken in a way that's correlated with the very metric you're testing. Any A/B test result "
    "should be checked for SRM before it's trusted, because a broken split can produce a "
    "statistically 'significant' result that has nothing to do with the treatment.",
)

with st.sidebar:
    st.header("⚙️ Observed Counts")
    n1 = st.number_input("Control users observed", value=51042, min_value=1, step=100)
    n2 = st.number_input("Variant users observed", value=49183, min_value=1, step=100)
    expected_ratio = st.slider("Intended Control ratio", 0.05, 0.95, 0.50, step=0.01)
    srm_alpha = st.select_slider("SRM significance threshold", options=[0.01, 0.001, 0.0001], value=0.001)

chi2, p_value = srm_chi_square(n1, n2, expected_ratio=expected_ratio)
total = n1 + n2
observed_ratio = n1 / total

c1, c2, c3, c4 = st.columns(4)
c1.metric("Observed Control ratio", f"{observed_ratio:.4f}")
c2.metric("Expected Control ratio", f"{expected_ratio:.4f}")
c3.metric("Chi-square statistic", f"{chi2:.3f}")
c4.metric("p-value", f"{p_value:.5f}")

st.divider()

if p_value < srm_alpha:
    st.error(
        f"🚨 **SRM detected.** The observed split deviates from the expected ratio far more than "
        f"chance would explain (p = {p_value:.5f} < {srm_alpha}). **Do not trust this experiment's "
        f"results** until the assignment/logging pipeline is investigated."
    )
else:
    st.success(
        f"✅ **No SRM detected.** The observed split is consistent with the intended ratio "
        f"(p = {p_value:.5f} ≥ {srm_alpha}). Safe to proceed with interpreting the experiment's metrics."
    )

st.subheader("1) Observed vs. Expected Split")
fig = go.Figure(
    data=[
        go.Bar(name="Observed", x=["Control", "Variant"], y=[n1, n2], marker_color="#636EFA"),
        go.Bar(
            name="Expected",
            x=["Control", "Variant"],
            y=[total * expected_ratio, total * (1 - expected_ratio)],
            marker_color="#CCCCCC",
        ),
    ]
)
fig.update_layout(barmode="group", height=380, yaxis_title="Users")
st.plotly_chart(fig, use_container_width=True)

st.subheader("2) Sensitivity: How Imbalanced Would Trigger SRM?")
imbalance_range = np.linspace(0.30, 0.70, 81)
p_values_range = [srm_chi_square(total * r, total * (1 - r), expected_ratio=expected_ratio)[1] for r in imbalance_range]
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=imbalance_range, y=p_values_range, mode="lines", line=dict(color="#00CC96", width=3)))
fig2.add_hline(y=srm_alpha, line_dash="dash", line_color="#EF553B", annotation_text=f"SRM threshold = {srm_alpha}")
fig2.add_vline(x=observed_ratio, line_dash="dot", line_color="gray", annotation_text="Your observed ratio")
fig2.update_layout(height=340, xaxis_title="Observed Control ratio", yaxis_title="p-value", yaxis_type="log")
st.plotly_chart(fig2, use_container_width=True)

st.markdown(
    """
    <div class="pitfall-card">
    <strong>Common real-world causes of SRM:</strong><br>
    • Bot traffic filtered differently between arms &nbsp;•&nbsp; Redirect/latency differences causing drop-off before assignment logs<br>
    • Caching serving a stale variant to some users &nbsp;•&nbsp; Client-side assignment failing silently on older app versions<br>
    • Server crashes or timeouts specific to one variant's code path
    </div>
    """,
    unsafe_allow_html=True,
)
