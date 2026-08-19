import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils.theme import inject_base_css, page_header, description_box
from utils.stats_engine import bonferroni_correction, benjamini_hochberg

st.set_page_config(page_title="Pitfalls Playbook", layout="wide")
inject_base_css()

page_header("Pitfalls Playbook", "The traps that quietly invalidate real experiments.", "📚")
description_box(
    "A field guide to the statistical mistakes that most often turn a well-designed experiment "
    "into a misleading result — with interactive demonstrations, not just definitions.",
    "Each card below covers a pitfall you're likely to encounter running real experiments, why it "
    "happens, and how to guard against it. Two of them — Simpson's Paradox and Multiple Testing "
    "— have interactive demos built in.",
)

# ------------------------------------------------------------------
# 1) Simpson's Paradox — interactive
# ------------------------------------------------------------------
st.subheader("1) Simpson's Paradox")
st.markdown(
    """
    <div class="pitfall-card">
    A treatment can look <strong>worse</strong> in every individual segment yet look
    <strong>better</strong> overall (or vice versa) once segments are combined — because the
    segments have different sizes and different baseline rates. Always check whether your topline
    result holds up within key segments (device, geography, new vs. returning users) before trusting it.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("▶ Interactive demo: build a Simpson's Paradox"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Segment A (e.g. Mobile)**")
        a_ctrl_n = st.number_input("Segment A — Control users", value=800, min_value=1, key="a_ctrl_n")
        a_ctrl_x = st.number_input("Segment A — Control conversions", value=40, min_value=0, key="a_ctrl_x")
        a_var_n = st.number_input("Segment A — Variant users", value=200, min_value=1, key="a_var_n")
        a_var_x = st.number_input("Segment A — Variant conversions", value=16, min_value=0, key="a_var_x")
    with col2:
        st.markdown("**Segment B (e.g. Desktop)**")
        b_ctrl_n = st.number_input("Segment B — Control users", value=200, min_value=1, key="b_ctrl_n")
        b_ctrl_x = st.number_input("Segment B — Control conversions", value=34, min_value=0, key="b_ctrl_x")
        b_var_n = st.number_input("Segment B — Variant users", value=800, min_value=1, key="b_var_n")
        b_var_x = st.number_input("Segment B — Variant conversions", value=120, min_value=0, key="b_var_x")

    a_ctrl_rate, a_var_rate = a_ctrl_x / a_ctrl_n, a_var_x / a_var_n
    b_ctrl_rate, b_var_rate = b_ctrl_x / b_ctrl_n, b_var_x / b_var_n
    total_ctrl_rate = (a_ctrl_x + b_ctrl_x) / (a_ctrl_n + b_ctrl_n)
    total_var_rate = (a_var_x + b_var_x) / (a_var_n + b_var_n)

    df = pd.DataFrame(
        {
            "Group": ["Segment A", "Segment B", "Overall (combined)"],
            "Control rate": [a_ctrl_rate, b_ctrl_rate, total_ctrl_rate],
            "Variant rate": [a_var_rate, b_var_rate, total_var_rate],
        }
    )
    df["Variant wins?"] = df["Variant rate"] > df["Control rate"]
    st.dataframe(
        df.style.format({"Control rate": "{:.2%}", "Variant rate": "{:.2%}"}),
        use_container_width=True,
        hide_index=True,
    )

    segment_agree = df["Variant wins?"].iloc[0] == df["Variant wins?"].iloc[1]
    overall_matches = df["Variant wins?"].iloc[2] == df["Variant wins?"].iloc[0]
    if segment_agree and not overall_matches:
        st.error(
            "⚠️ Simpson's Paradox triggered: both segments agree with each other, but the "
            "**combined result flips direction** — driven by the segments having very different "
            "sizes and baseline rates."
        )
    else:
        st.info("Adjust the numbers above (try giving each segment a very different size and baseline rate) to trigger the paradox.")

st.divider()

# ------------------------------------------------------------------
# 2) Multiple Testing — interactive
# ------------------------------------------------------------------
st.subheader("2) Multiple Testing / Multiple Comparisons")
st.markdown(
    """
    <div class="pitfall-card">
    Testing many metrics or many variants at once inflates the chance that <em>something</em> looks
    significant purely by chance. With 20 independent metrics tested at α = 0.05, you expect
    roughly one false positive even if nothing actually changed.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("▶ Interactive demo: correction methods"):
    num_metrics = st.slider("Number of metrics/variants tested", 2, 50, 20)
    seed_mt = st.number_input("Seed", value=7, key="mt_seed")
    rng_mt = np.random.default_rng(int(seed_mt))
    # Simulate p-values under the null (uniform) plus a couple of true small effects
    p_values = rng_mt.uniform(0, 1, size=num_metrics)
    n_true_effects = max(0, num_metrics // 10)
    if n_true_effects > 0:
        p_values[:n_true_effects] = rng_mt.uniform(0, 0.01, size=n_true_effects)

    alpha_mt = st.slider("α", 0.01, 0.20, 0.05, key="mt_alpha")
    adj_alpha, bonf_reject = bonferroni_correction(p_values, alpha=alpha_mt)
    bh_reject = benjamini_hochberg(p_values, alpha=alpha_mt)

    raw_sig = int(np.sum(p_values <= alpha_mt))
    bonf_sig = int(np.sum(bonf_reject))
    bh_sig = int(np.sum(bh_reject))

    c1, c2, c3 = st.columns(3)
    c1.metric("Flagged significant — no correction", raw_sig)
    c2.metric("Flagged significant — Bonferroni", bonf_sig)
    c3.metric("Flagged significant — Benjamini-Hochberg", bh_sig)

    df_mt = pd.DataFrame(
        {
            "Metric": [f"Metric {i+1}" for i in range(num_metrics)],
            "p-value": p_values,
            "Raw significant?": p_values <= alpha_mt,
            "Bonferroni significant?": bonf_reject,
            "BH significant?": bh_reject,
        }
    ).sort_values("p-value")
    st.dataframe(df_mt.style.format({"p-value": "{:.4f}"}), use_container_width=True, hide_index=True)
    st.caption(
        f"Bonferroni controls family-wise error rate (strict, adjusted α = {adj_alpha:.4f}); "
        "Benjamini-Hochberg controls false discovery rate (less conservative, better when testing many metrics)."
    )

st.divider()

# ------------------------------------------------------------------
# 3) Static pitfall cards
# ------------------------------------------------------------------
st.subheader("3) More Pitfalls to Watch For")

pitfalls = [
    (
        "Novelty & Primacy Effects",
        "Users react to *any* change at first — novelty can inflate early lift, or primacy/change "
        "aversion can suppress it — before settling to a steady state. Always plot the metric by "
        "day-since-exposure, not just the pooled average, and prefer experienced users' data.",
    ),
    (
        "Peeking and Optional Stopping",
        "Stopping a test the moment it crosses significance inflates the true false-positive rate "
        "far above your nominal α — see the dedicated Sequential Testing lab for a live simulation.",
    ),
    (
        "Interference Between Users (Network Effects)",
        "If Variant users can influence Control users' experience (marketplaces, social feeds, "
        "shared inventory), the 'independent units' assumption breaks and effects can be biased "
        "in either direction. Cluster-level or geo-based randomization is often the fix.",
    ),
    (
        "Metric Choice Mismatches Business Goal",
        "Optimizing for a proxy metric (clicks) that doesn't track the true goal (long-term revenue "
        "or retention) can ship changes that win the test and quietly hurt the business.",
    ),
    (
        "Underpowered Segments",
        "An overall test can be well-powered while every individual segment you later slice by "
        "(country, device, cohort) is statistically noise — segment-level 'insights' after the "
        "fact are usually overfitting unless pre-registered and powered for.",
    ),
]

cols = st.columns(2)
for i, (title, body) in enumerate(pitfalls):
    with cols[i % 2]:
        st.markdown(
            f"""
            <div class="pitfall-card">
                <strong>{title}</strong><br><br>
                <span class="explanation-text">{body}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
