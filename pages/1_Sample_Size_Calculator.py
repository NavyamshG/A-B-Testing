import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from utils.theme import inject_base_css, page_header, description_box, GOOD, BAD, NEUTRAL
from utils.stats_engine import sample_size_two_proportions, power_for_n

st.set_page_config(page_title="Sample Size Calculator", layout="wide")
inject_base_css()

page_header("Sample Size Calculator", "Design your experiment before you run it.", "📐")
description_box(
    "Estimate the sample size required to detect a given effect size at a chosen significance "
    "level and power, and visualize the trade-off between sample size, effect size, and power.",
    "Under-powered experiments are one of the most common reasons real A/B tests produce "
    "misleading results. This page uses the standard normal-approximation formula for a "
    "two-proportion test so you can plan sample size, or check how much power you actually have "
    "given a fixed traffic budget.",
)

with st.sidebar:
    st.header("⚙️ Design Inputs")
    baseline = st.slider("Baseline conversion rate (Control)", 0.001, 0.5, 0.10, step=0.001, format="%.3f")
    mde_pct = st.slider("Minimum Detectable Effect (relative %)", 1, 100, 15, step=1)
    mde = baseline * (mde_pct / 100)
    alpha = st.slider("Significance level (α)", 0.01, 0.20, 0.05, step=0.01)
    power = st.slider("Desired power (1 − β)", 0.5, 0.99, 0.80, step=0.01)
    two_sided = st.toggle("Two-sided test", value=True)

n_required = sample_size_two_proportions(baseline, mde, alpha=alpha, power=power, two_sided=two_sided)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Required N per group", f"{n_required:,}")
c2.metric("Total sample size", f"{2 * n_required:,}")
c3.metric("Absolute MDE", f"{mde:.3%}")
c4.metric("Target CTR (Variant)", f"{baseline + mde:.3%}")

st.divider()

st.subheader("1) Sample Size vs. Minimum Detectable Effect")
mde_range_pct = np.linspace(2, 100, 60)
n_curve = [
    sample_size_two_proportions(baseline, baseline * (m / 100), alpha=alpha, power=power, two_sided=two_sided)
    for m in mde_range_pct
]
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=mde_range_pct, y=n_curve, mode="lines", line=dict(color=NEUTRAL, width=3)))
fig1.add_vline(x=mde_pct, line_dash="dash", line_color=BAD)
fig1.update_layout(
    height=380,
    xaxis_title="MDE (relative %)",
    yaxis_title="Required sample size per group",
    margin=dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(fig1, use_container_width=True)
st.caption("Smaller effects require dramatically more traffic — this is why chasing tiny lifts is expensive.")

st.divider()

st.subheader("2) Power Given a Fixed Traffic Budget")
n_budget = st.number_input(
    "If you only have this many users per group, what power do you actually get?",
    min_value=10,
    value=int(min(n_required, 5000)),
    step=100,
)
achieved_power = power_for_n(baseline, mde, n_budget, alpha=alpha, two_sided=two_sided)

n_range = np.linspace(max(10, n_budget * 0.1), n_budget * 2.5, 60)
power_curve = [power_for_n(baseline, mde, n, alpha=alpha, two_sided=two_sided) for n in n_range]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=n_range, y=power_curve, mode="lines", line=dict(color=GOOD, width=3)))
fig2.add_hline(y=0.8, line_dash="dot", line_color="gray", annotation_text="80% power")
fig2.add_vline(x=n_budget, line_dash="dash", line_color=BAD)
fig2.update_layout(
    height=380,
    xaxis_title="Sample size per group",
    yaxis_title="Achieved power",
    yaxis_range=[0, 1],
    margin=dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(fig2, use_container_width=True)

if achieved_power >= power:
    st.success(f"At n={n_budget:,} per group, achieved power is **{achieved_power:.1%}** — sufficient for your target of {power:.0%}.")
else:
    shortfall = n_required - n_budget
    st.warning(
        f"At n={n_budget:,} per group, achieved power is only **{achieved_power:.1%}** "
        f"— below your {power:.0%} target. You'd need about {max(shortfall, 0):,} more users per group."
    )

st.markdown(
    """
    <div class="callout">
    ✨ <strong>Try this:</strong> Set baseline to 5%, MDE to 10%, and watch the required sample size
    climb into the tens of thousands — a realistic picture of why small-effect experiments at
    low-traffic companies can take months to reach significance.
    </div>
    """,
    unsafe_allow_html=True,
)
