import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import math

from utils.theme import inject_base_css, page_header, description_box, GOOD, BAD, PURPLE
from utils.stats_engine import (
    validate_counts,
    two_prop_ztest,
    welch_ttest_bernoulli,
    fishers_exact,
    wald_ci_diff,
    newcombe_ci_diff,
    pct,
)

st.set_page_config(page_title="Frequentist Lab", layout="wide")
inject_base_css()

page_header("Frequentist Inference Lab", "How different tests read the same click data.", "📊")
description_box(
    "Evaluate p-value and confidence interval variance across Bernoulli and Binomial modeling "
    "frameworks to compare the sensitivity of the Z-test, Welch's T-test, and Fisher's Exact test.",
    "This lab demonstrates how different mathematical 'lenses' interpret the same data, comparing "
    "the normal-approximation Z and T-tests against the exact probability calculations of Fisher's "
    "method. It reveals whether a statistical 'win' is a robust result or simply a byproduct of the "
    "specific distribution and test selected.",
)

with st.sidebar:
    st.header("🕹️ Scenario Selector")
    scenario = st.selectbox("Choose a scenario:", ["Manual Entry", "Small Sample", "Marginal Win", "Clear Winner"])
    presets = {
        "Manual Entry": (20, 200, 35, 200),
        "Small Sample": (2, 20, 5, 20),
        "Marginal Win": (100, 1000, 125, 1000),
        "Clear Winner": (50, 1000, 120, 1000),
    }
    def_x1, def_n1, def_x2, def_n2 = presets[scenario]

    st.divider()
    x1 = st.number_input("Control Clicks", value=def_x1, min_value=0)
    n1 = st.number_input("Control Views", value=def_n1, min_value=1)
    x2 = st.number_input("Variant Clicks", value=def_x2, min_value=0)
    n2 = st.number_input("Variant Views", value=def_n2, min_value=1)
    alpha = st.slider("Significance α", 0.01, 0.20, 0.05)

for err in [validate_counts(int(x1), int(n1), "Control"), validate_counts(int(x2), int(n2), "Variant")]:
    if err:
        st.error(err)
        st.stop()

p1, p2 = x1 / n1, x2 / n2
diff = p2 - p1

z_stat, z_p = two_prop_ztest(x1, n1, x2, n2)
t_stat, t_df, t_p = welch_ttest_bernoulli(x1, n1, x2, n2)
odds, f_p = fishers_exact(x1, n1, x2, n2)

tests = {"z-test": z_p, "t-test": t_p, "Fisher": f_p}
wins = [name for name, p in tests.items() if np.isfinite(p) and p <= alpha]
losses = [name for name, p in tests.items() if np.isfinite(p) and p > alpha]

wald_d, wald_lo, wald_hi = wald_ci_diff(x1, n1, x2, n2, alpha=alpha)
newc_d, newc_lo, newc_hi = newcombe_ci_diff(x1, n1, x2, n2, alpha=alpha)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Control CTR", pct(p1))
c2.metric("Variant CTR", pct(p2))
c3.metric("Δ CTR (Abs)", pct(diff))
c4.metric("Confidence (Z)", pct(1 - z_p) if np.isfinite(z_p) else "NA")

st.divider()

st.subheader("1) Comparative Statistical Verdicts")
col_plot, col_results = st.columns([1.5, 1])

with col_plot:
    x_axis = np.linspace(max(0, min(p1, p2) - 0.15), min(1, max(p1, p2) + 0.15), 500)
    y1 = np.exp(-0.5 * ((x_axis - p1) / math.sqrt(p1 * (1 - p1) / n1)) ** 2) / (
        math.sqrt(2 * math.pi) * math.sqrt(p1 * (1 - p1) / n1)
    )
    y2 = np.exp(-0.5 * ((x_axis - p2) / math.sqrt(p2 * (1 - p2) / n2)) ** 2) / (
        math.sqrt(2 * math.pi) * math.sqrt(p2 * (1 - p2) / n2)
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_axis, y=y1, fill="tozeroy", name="Control", line_color="#636EFA"))
    fig.add_trace(go.Scatter(x=x_axis, y=y2, fill="tozeroy", name="Variant", line_color="#00CC96"))
    fig.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

with col_results:
    if wins:
        st.success(f"**Winner according to:** {', '.join(wins)}")
    if losses:
        st.error(f"**No Significance according to:** {', '.join(losses)}")

    st.markdown("**p-value Comparison**")
    fig_p = go.Figure(
        go.Bar(
            x=list(tests.keys()),
            y=list(tests.values()),
            marker_color=[GOOD if p <= alpha else BAD for p in tests.values()],
            text=[f"{p:.4f}" for p in tests.values()],
            textposition="auto",
        )
    )
    fig_p.add_hline(y=alpha, line_dash="dash")
    fig_p.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_p, use_container_width=True)

st.markdown(
    """
    <div class="callout">
    ✨ <strong>To see a magic:</strong> Set Control clicks: <strong>20</strong>, Control views: <strong>200</strong>,
    Variant clicks: <strong>35</strong>, Variant views: <strong>200</strong>, and Significance level: <strong>0.04</strong>
    — watch the three tests disagree on whether this is a "win."
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

st.subheader("2) Delta Confidence Intervals")
st.markdown(
    f"""<span style="font-size: 1.15rem;"><strong>Wald Interval (Binomial):</strong>
    <code>{pct(wald_lo)}</code> to <code>{pct(wald_hi)}</code></span>""",
    unsafe_allow_html=True,
)
st.markdown(
    f"""<span style="font-size: 1.15rem;"><strong>Newcombe Interval (Wilson):</strong>
    <code>{pct(newc_lo)}</code> to <code>{pct(newc_hi)}</code></span>""",
    unsafe_allow_html=True,
)

fig_ci = go.Figure()
fig_ci.add_trace(
    go.Scatter(
        x=[diff], y=["Wald (Binomial)"], mode="markers",
        error_x=dict(type="data", array=[wald_hi - diff], arrayminus=[diff - wald_lo], visible=True),
        marker=dict(size=12, color=PURPLE),
    )
)
fig_ci.add_trace(
    go.Scatter(
        x=[diff], y=["Newcombe (Wilson)"], mode="markers",
        error_x=dict(type="data", array=[newc_hi - diff], arrayminus=[diff - newc_lo], visible=True),
        marker=dict(size=12, color=BAD),
    )
)
fig_ci.add_vline(x=0, line_dash="dash", line_color="gray")
fig_ci.update_layout(height=280, xaxis_title="Abs Difference", showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig_ci, use_container_width=True)

st.info("Looking for the peeking simulation? It now has its own dedicated page — **Sequential Testing** — with always-valid p-values added.")
