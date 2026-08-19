import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go

from utils.theme import inject_base_css, page_header, description_box
from utils.stats_engine import validate_counts, beta_credible_interval, expected_loss

st.set_page_config(page_title="Bayesian Lab", layout="wide")
inject_base_css()

page_header("Bayesian CTR Inference", "Beta-Binomial posteriors, credible intervals, and decision-ready probabilities.", "🧠")
description_box(
    "Model Control and Variant CTR as Beta-Binomial posteriors, and derive the probability that "
    "Variant beats Control along with the expected loss of each decision.",
    "Unlike p-values, Bayesian inference gives you a direct probability statement — "
    "'there is an 87% chance Variant is better' — plus an expected-loss framework for deciding "
    "whether to ship even without full certainty. Adjust the prior and likelihood weight in the "
    "sidebar to see how beliefs and evidence shape the posterior.",
)

with st.sidebar:
    st.header("🕹️ Data")
    scenario = st.selectbox("Choose a scenario:", ["Manual Entry", "Small Sample", "Marginal Win", "Clear Winner"])
    presets = {
        "Manual Entry": (20, 200, 35, 200),
        "Small Sample": (2, 20, 5, 20),
        "Marginal Win": (100, 1000, 125, 1000),
        "Clear Winner": (50, 1000, 120, 1000),
    }
    def_x1, def_n1, def_x2, def_n2 = presets[scenario]
    x1 = st.number_input("Control Clicks", value=def_x1, min_value=0)
    n1 = st.number_input("Control Views", value=def_n1, min_value=1)
    x2 = st.number_input("Variant Clicks", value=def_x2, min_value=0)
    n2 = st.number_input("Variant Views", value=def_n2, min_value=1)
    seed = st.number_input("Seed", value=42)

    st.divider()
    st.header("🧠 Bayesian Settings")
    with st.expander("Prior Setup", expanded=True):
        prior_mode = st.selectbox("Choose prior type:", ["Uniform (Beta(1,1))", "Jeffreys (Beta(0.5,0.5))", "Custom Beta(a,b)"])
        if prior_mode == "Custom Beta(a,b)":
            prior_a = st.number_input("Prior α (a)", min_value=0.01, value=1.0, step=0.5)
            prior_b = st.number_input("Prior β (b)", min_value=0.01, value=1.0, step=0.5)
        elif prior_mode == "Jeffreys (Beta(0.5,0.5))":
            prior_a, prior_b = 0.5, 0.5
        else:
            prior_a, prior_b = 1.0, 1.0
        st.caption("Tip: Stronger priors = larger (a+b). Uniform/Jeffreys are weak, neutral priors.")

    with st.expander("Posterior Sampling & Interval"):
        bayes_samples = st.slider("Posterior samples", 5000, 100000, 20000, step=5000)
        credible_level = st.slider("Credible interval level", 0.80, 0.99, 0.95, step=0.01)
        show_prior_overlay = st.toggle("Show prior vs posterior overlay", value=True)

    with st.expander("Likelihood Strength Demo"):
        st.caption("Down/up-weight the data while keeping the same observed CTR.")
        like_weight = st.slider("Likelihood weight (0.1 = weak, 1.0 = actual, 3.0 = strong)", 0.1, 3.0, 1.0, step=0.1)

    with st.expander("Decision Threshold"):
        loss_threshold = st.slider(
            "Max acceptable expected loss to ship Variant", 0.0001, 0.05, 0.005, step=0.0001, format="%.4f"
        )

for err in [validate_counts(int(x1), int(n1), "Control"), validate_counts(int(x2), int(n2), "Variant")]:
    if err:
        st.error(err)
        st.stop()

a0, b0 = float(prior_a), float(prior_b)
x1w = min(max(0, int(round(float(x1) * like_weight))), int(round(float(n1) * like_weight)) or 1)
n1w = max(1, int(round(float(n1) * like_weight)))
x2w = min(max(0, int(round(float(x2) * like_weight))), int(round(float(n2) * like_weight)) or 1)
n2w = max(1, int(round(float(n2) * like_weight)))
x1w, x2w = min(x1w, n1w), min(x2w, n2w)

a1_post, b1_post = a0 + x1w, b0 + (n1w - x1w)
a2_post, b2_post = a0 + x2w, b0 + (n2w - x2w)

rng_b = np.random.default_rng(int(seed) + 7)
control_samples = rng_b.beta(a1_post, b1_post, size=int(bayes_samples))
variant_samples = rng_b.beta(a2_post, b2_post, size=int(bayes_samples))
lift_samples = variant_samples - control_samples

ctrl_lo, ctrl_hi = beta_credible_interval(control_samples, level=float(credible_level))
var_lo, var_hi = beta_credible_interval(variant_samples, level=float(credible_level))
lift_lo, lift_hi = beta_credible_interval(lift_samples, level=float(credible_level))
prob_variant_wins = float(np.mean(variant_samples > control_samples))
loss_choose_variant, loss_choose_control = expected_loss(control_samples, variant_samples)

k1, k2, k3, k4 = st.columns(4)
k1.metric("P(Variant > Control)", f"{prob_variant_wins:.2%}")
k2.metric(f"Control {int(credible_level*100)}% CrI", f"{ctrl_lo:.2%} — {ctrl_hi:.2%}")
k3.metric(f"Variant {int(credible_level*100)}% CrI", f"{var_lo:.2%} — {var_hi:.2%}")
k4.metric(f"Lift {int(credible_level*100)}% CrI", f"{lift_lo:.2%} — {lift_hi:.2%}")

st.divider()

st.subheader("1) Posterior Distributions")
x_max = max(0.25, float(max(x1 / n1, x2 / n2) * 3))
xs = np.linspace(0, min(1.0, x_max), 700)
ctrl_pdf = stats.beta.pdf(xs, a1_post, b1_post)
var_pdf = stats.beta.pdf(xs, a2_post, b2_post)

fig_post = go.Figure()
fig_post.add_trace(go.Scatter(x=xs, y=ctrl_pdf, fill="tozeroy", name="Control Posterior", line_color="#636EFA"))
fig_post.add_trace(go.Scatter(x=xs, y=var_pdf, fill="tozeroy", name="Variant Posterior", line_color="#00CC96"))
if show_prior_overlay:
    prior_pdf = stats.beta.pdf(xs, a0, b0)
    fig_post.add_trace(go.Scatter(x=xs, y=prior_pdf, name=f"Prior Beta({a0:g},{b0:g})", line=dict(color="gray", dash="dash")))
fig_post.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="CTR", yaxis_title="Posterior Density")
st.plotly_chart(fig_post, use_container_width=True)

st.markdown(
    f"""
- **Prior:** Beta({a0:g}, {b0:g})
- **Likelihood weight:** {like_weight:.1f}× (effective views: Control={n1w}, Variant={n2w})
- **Posterior (Control):** Beta({a1_post:.1f}, {b1_post:.1f})
- **Posterior (Variant):** Beta({a2_post:.1f}, {b2_post:.1f})
""".strip()
)

st.divider()

st.subheader("2) Lift Posterior & Credible Interval")
lift_mean = float(np.mean(lift_samples))
fig_bci = go.Figure()
fig_bci.add_trace(
    go.Scatter(
        x=[lift_mean], y=[f"Lift ({int(credible_level*100)}% CrI)"], mode="markers",
        error_x=dict(type="data", array=[lift_hi - lift_mean], arrayminus=[lift_mean - lift_lo], visible=True),
        marker=dict(size=12, color="#FF4B4B"),
    )
)
fig_bci.add_vline(x=0, line_dash="dash", line_color="gray")
fig_bci.update_layout(height=220, xaxis_title="Abs Lift", showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig_bci, use_container_width=True)

fig_lift = go.Figure()
fig_lift.add_trace(go.Histogram(x=lift_samples, nbinsx=70, marker_color="#AB63FA"))
fig_lift.add_vline(x=0, line_dash="dash", line_color="gray")
fig_lift.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Absolute Lift", yaxis_title="Frequency")
st.plotly_chart(fig_lift, use_container_width=True)

st.divider()

st.subheader("3) Decision Support: Expected Loss")
st.caption(
    "Expected loss answers 'if I ship the wrong variant, how much CTR do I expect to give up?' "
    "It lets you ship confidently even before P(Variant > Control) reaches 95%."
)
l1, l2 = st.columns(2)
l1.metric("Expected loss if you ship Variant", f"{loss_choose_variant:.4%}")
l2.metric("Expected loss if you keep Control", f"{loss_choose_control:.4%}")

if loss_choose_variant <= loss_threshold:
    st.success(
        f"Expected loss from shipping Variant ({loss_choose_variant:.4%}) is within your "
        f"threshold ({loss_threshold:.4%}) — safe to ship even if P(win) isn't at 95%."
    )
else:
    st.warning(
        f"Expected loss from shipping Variant ({loss_choose_variant:.4%}) exceeds your "
        f"threshold ({loss_threshold:.4%}) — collect more data before deciding."
    )
