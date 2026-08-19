import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import scipy.stats as stats

from utils.theme import inject_base_css, page_header, description_box
from utils.stats_engine import cuped_adjust

st.set_page_config(page_title="CUPED Lab", layout="wide")
inject_base_css()

page_header("CUPED Variance Reduction", "Use pre-experiment data to see effects more clearly.", "🎛️")
description_box(
    "Simulate a metric with pre-experiment (covariate) and post-experiment (outcome) values per "
    "user, apply CUPED adjustment, and compare statistical power before vs. after.",
    "CUPED (Controlled-experiment Using Pre-Experiment Data), popularized by Microsoft's "
    "experimentation platform, removes the portion of outcome variance explained by a "
    "pre-experiment covariate — like a user's historical spend or engagement — without touching "
    "the treatment effect itself. The result: the same sample size detects smaller effects, or "
    "the same effect reaches significance with less traffic.",
)

with st.sidebar:
    st.header("⚙️ Simulation Settings")
    n_per_group = st.slider("Users per group", 200, 20000, 3000, step=200)
    true_effect = st.slider("True treatment effect (added to outcome)", 0.0, 5.0, 1.0, step=0.1)
    corr = st.slider("Correlation between pre-period and outcome metric", 0.0, 0.95, 0.6, step=0.05)
    noise_sd = st.slider("Outcome noise (std dev)", 1.0, 20.0, 10.0, step=0.5)
    seed = st.number_input("Seed", value=42)
    alpha = st.slider("Significance α", 0.01, 0.20, 0.05)

rng = np.random.default_rng(int(seed))

# Simulate a pre-period covariate (e.g. past spend/engagement), shared distribution for both groups
pre_control = rng.normal(50, 15, size=n_per_group)
pre_variant = rng.normal(50, 15, size=n_per_group)

# Outcome correlated with pre-period covariate, plus noise, plus true treatment effect on variant
def make_outcome(pre, effect, corr, noise_sd, rng):
    base_noise = rng.normal(0, noise_sd, size=len(pre))
    outcome = corr * (pre - pre.mean()) + np.sqrt(max(1e-9, 1 - corr**2)) * base_noise + 50
    return outcome + effect

outcome_control = make_outcome(pre_control, 0.0, corr, noise_sd, rng)
outcome_variant = make_outcome(pre_variant, true_effect, corr, noise_sd, rng)

# CUPED adjustment (theta estimated jointly across both groups' pre-period vs outcome)
all_pre = np.concatenate([pre_control, pre_variant])
all_outcome = np.concatenate([outcome_control, outcome_variant])
_, theta, _ = cuped_adjust(all_outcome, all_pre)

adj_control, _, var_reduction_c = cuped_adjust(outcome_control, pre_control, theta=theta)
adj_variant, _, var_reduction_v = cuped_adjust(outcome_variant, pre_variant, theta=theta)

# Raw t-test
t_raw, p_raw = stats.ttest_ind(outcome_variant, outcome_control, equal_var=False)
# CUPED-adjusted t-test
t_adj, p_adj = stats.ttest_ind(adj_variant, adj_control, equal_var=False)

var_before = np.var(np.concatenate([outcome_control, outcome_variant]), ddof=1)
var_after = np.var(np.concatenate([adj_control, adj_variant]), ddof=1)
variance_reduction_pct = 100 * (1 - var_after / var_before) if var_before > 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Variance reduction", f"{variance_reduction_pct:.1f}%")
c2.metric("Raw p-value", f"{p_raw:.4f}", delta="significant" if p_raw <= alpha else "not significant", delta_color="normal" if p_raw <= alpha else "off")
c3.metric("CUPED-adjusted p-value", f"{p_adj:.4f}", delta="significant" if p_adj <= alpha else "not significant", delta_color="normal" if p_adj <= alpha else "off")
c4.metric("CUPED θ (estimated)", f"{theta:.3f}")

st.divider()

st.subheader("1) Outcome Distributions: Raw vs. CUPED-Adjusted")
col1, col2 = st.columns(2)
with col1:
    fig_raw = go.Figure()
    fig_raw.add_trace(go.Histogram(x=outcome_control, name="Control", opacity=0.6, marker_color="#636EFA"))
    fig_raw.add_trace(go.Histogram(x=outcome_variant, name="Variant", opacity=0.6, marker_color="#00CC96"))
    fig_raw.update_layout(barmode="overlay", height=350, title="Raw Outcome", margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_raw, use_container_width=True)
with col2:
    fig_adj = go.Figure()
    fig_adj.add_trace(go.Histogram(x=adj_control, name="Control (adj.)", opacity=0.6, marker_color="#636EFA"))
    fig_adj.add_trace(go.Histogram(x=adj_variant, name="Variant (adj.)", opacity=0.6, marker_color="#00CC96"))
    fig_adj.update_layout(barmode="overlay", height=350, title="CUPED-Adjusted Outcome", margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_adj, use_container_width=True)

st.caption("Narrower, more separated distributions after adjustment make the same true effect easier to detect statistically.")

st.divider()

st.subheader("2) Effective Sample Size Savings")
if var_before > 0 and var_after > 0:
    effective_n_multiplier = var_before / var_after
    st.markdown(
        f"""
        A **{variance_reduction_pct:.1f}%** reduction in outcome variance means CUPED gives you the
        statistical power equivalent of roughly **{effective_n_multiplier:.2f}x** the sample size
        — without collecting a single additional user.
        """
    )
    fig_savings = go.Figure(
        go.Bar(
            x=["Raw sample size needed", "Effective sample size with CUPED"],
            y=[n_per_group, n_per_group / effective_n_multiplier],
            marker_color=["#EF553B", "#00CC96"],
            text=[f"{n_per_group:,}", f"{n_per_group/effective_n_multiplier:,.0f}"],
            textposition="auto",
        )
    )
    fig_savings.update_layout(height=320, yaxis_title="Users per group")
    st.plotly_chart(fig_savings, use_container_width=True)

st.markdown(
    """
    <div class="callout">
    ✨ <strong>Try this:</strong> Push the pre/post correlation slider to 0.9 — with a strongly
    predictive pre-period covariate, CUPED can cut required sample size by more than half.
    At correlation 0, CUPED does nothing, as expected.
    </div>
    """,
    unsafe_allow_html=True,
)
