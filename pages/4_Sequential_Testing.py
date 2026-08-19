import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from utils.theme import inject_base_css, page_header, description_box
from utils.stats_engine import two_prop_ztest, mixture_sequential_p_value

st.set_page_config(page_title="Sequential Testing", layout="wide")
inject_base_css()

page_header("Sequential Testing", "The danger of peeking — and how to peek safely.", "⏱️")
description_box(
    "Simulate repeatedly checking a fixed-horizon test as data accumulates, and compare the "
    "inflated false-positive rate of a naive fixed-alpha p-value against an always-valid "
    "sequential p-value that stays valid under continuous monitoring.",
    "Checking a test's p-value every day and stopping the first time it crosses 0.05 is one of "
    "the most common ways real experiments produce false wins. This page simulates that behavior "
    "under the null hypothesis (no true difference) to show how badly the error rate inflates, "
    "and demonstrates a mixture-based always-valid p-value (mSPRT-style) that controls the false "
    "positive rate even when you look continuously.",
)

with st.sidebar:
    st.header("⚙️ Simulation Settings")
    base_rate = st.slider("True CTR (same for both groups — null is true)", 0.01, 0.5, 0.10, step=0.01)
    n_total = st.number_input("Total sample size per group (end of test)", value=2000, min_value=100, step=100)
    looks = st.slider("Number of looks (k)", 2, 40, 15)
    sims = st.slider("Simulations", 100, 3000, 500, step=100)
    alpha = st.slider("Nominal α", 0.01, 0.20, 0.05)
    tau = st.slider("Mixture prior width (τ) for always-valid test", 0.01, 1.0, 0.1, step=0.01)
    seed = st.number_input("Seed", value=42)

rng = np.random.default_rng(int(seed))
sample_points = np.linspace(max(10, n_total // looks), n_total, looks).astype(int)

# Single example journey for the chart
a_hits = rng.binomial(1, base_rate, size=int(n_total))
b_hits = rng.binomial(1, base_rate, size=int(n_total))
naive_journey, valid_journey = [], []
for n_pt in sample_points:
    xa, xb = a_hits[:n_pt].sum(), b_hits[:n_pt].sum()
    z, p_naive = two_prop_ztest(xa, n_pt, xb, n_pt)
    p_valid = mixture_sequential_p_value(z, n_pt, tau=tau)
    naive_journey.append(p_naive)
    valid_journey.append(p_valid)

st.subheader("1) One Experiment's p-value Journey")
fig_journey = go.Figure()
fig_journey.add_trace(go.Scatter(x=sample_points, y=naive_journey, mode="lines+markers", name="Naive fixed-α p-value", line_color="#EF553B"))
fig_journey.add_trace(go.Scatter(x=sample_points, y=valid_journey, mode="lines+markers", name="Always-valid p-value", line_color="#00CC96"))
fig_journey.add_hline(y=alpha, line_dash="dash", annotation_text=f"α = {alpha}")
fig_journey.update_layout(height=380, yaxis_title="p-value", yaxis_range=[0, 1], xaxis_title="Cumulative sample size per group")
st.plotly_chart(fig_journey, use_container_width=True)
st.caption(
    "The null hypothesis is true here (both groups share the same true CTR), so any 'significant' "
    "crossing below the dashed line is a false positive by construction."
)

st.divider()

st.subheader("2) False Positive Rate Under Repeated Peeking")
run_sim = st.button("▶ Run Monte Carlo Simulation", type="primary")

if run_sim:
    naive_fp, valid_fp = 0, 0
    progress = st.progress(0, text="Simulating...")
    for i in range(int(sims)):
        a = rng.binomial(1, base_rate, size=int(n_total))
        b = rng.binomial(1, base_rate, size=int(n_total))
        naive_flagged, valid_flagged = False, False
        for n_pt in sample_points:
            xa, xb = a[:n_pt].sum(), b[:n_pt].sum()
            z, p_naive = two_prop_ztest(xa, n_pt, xb, n_pt)
            p_valid = mixture_sequential_p_value(z, n_pt, tau=tau)
            if p_naive <= alpha:
                naive_flagged = True
            if p_valid <= alpha:
                valid_flagged = True
        naive_fp += naive_flagged
        valid_fp += valid_flagged
        if i % max(1, int(sims) // 20) == 0:
            progress.progress(min(1.0, i / sims), text=f"Simulating... {i}/{int(sims)}")
    progress.empty()

    naive_rate = naive_fp / sims
    valid_rate = valid_fp / sims

    c1, c2, c3 = st.columns(3)
    c1.metric("Naive false positive rate", f"{naive_rate:.1%}", delta=f"{(naive_rate/alpha):.1f}x nominal α", delta_color="inverse")
    c2.metric("Always-valid false positive rate", f"{valid_rate:.1%}", delta=f"{(valid_rate/alpha):.1f}x nominal α", delta_color="inverse")
    c3.metric("Nominal α (target)", f"{alpha:.1%}")

    fig_bar = go.Figure(
        go.Bar(
            x=["Naive (peek & stop)", "Always-valid (mSPRT)", "Nominal target"],
            y=[naive_rate, valid_rate, alpha],
            marker_color=["#EF553B", "#00CC96", "#636EFA"],
            text=[f"{naive_rate:.1%}", f"{valid_rate:.1%}", f"{alpha:.1%}"],
            textposition="auto",
        )
    )
    fig_bar.update_layout(height=340, yaxis_title="False positive rate")
    st.plotly_chart(fig_bar, use_container_width=True)

    if naive_rate > alpha * 1.3:
        st.error(
            f"Checking {looks} times with a naive p-value inflated your true error rate to "
            f"{naive_rate:.1%} — roughly {naive_rate/alpha:.1f}x what you signed up for."
        )
    if valid_rate <= alpha * 1.3:
        st.success("The always-valid p-value stayed close to the nominal α even under continuous peeking.")
else:
    st.info("Click **Run Monte Carlo Simulation** to compute false positive rates across many simulated experiments.")

st.markdown(
    """
    <div class="callout">
    ✨ <strong>Try this:</strong> Push looks to 30+ and sims to 2000 — the naive false positive rate
    climbs well past 20% even though every individual test was run at α = 0.05. The always-valid
    test barely moves.
    </div>
    """,
    unsafe_allow_html=True,
)
