import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import streamlit as st
from utils.theme import inject_base_css, page_header

st.set_page_config(
    page_title="A/B Testing Studio",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_base_css()

page_header(
    "A/B Testing Studio",
    "An interactive, end-to-end laboratory for designing, running, and correctly interpreting A/B tests.",
    emoji="🧪",
)

st.markdown(
    """
    <div class="description-box">
        <strong>What this is:</strong> a hands-on toolkit that walks through the full lifecycle
        of an experiment — from deciding <em>how much data you need</em>, to comparing
        <em>frequentist vs. Bayesian</em> inference on the same data, to the statistical traps
        (peeking, sample ratio mismatch, multiple comparisons) that quietly invalidate results
        in real companies.
        <hr style="margin: 15px 0; border: 0; border-top: 1px solid #ddd;">
        <div class="explanation-text">
            Every page is interactive — change the inputs and watch the math respond in real time.
            Built to demonstrate both software engineering craft and applied experimentation know-how.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("Explore the labs")

cards = [
    ("📐", "Sample Size Calculator", "Design your experiment before you run it: required sample size, power curves, and MDE trade-offs.", "pages/1_Sample_Size_Calculator.py"),
    ("📊", "Frequentist Lab", "Compare Z-test, Welch's T-test, and Fisher's Exact on the same click data — see when they disagree.", "pages/2_Frequentist_Lab.py"),
    ("🧠", "Bayesian Lab", "Beta-Binomial posteriors, credible intervals, probability-to-win, and expected loss.", "pages/3_Bayesian_Lab.py"),
    ("⏱️", "Sequential Testing", "Why peeking inflates false positives, and how always-valid p-values fix it.", "pages/4_Sequential_Testing.py"),
    ("🎛️", "CUPED Variance Reduction", "Use pre-experiment data to shrink variance and detect smaller effects with the same sample.", "pages/5_CUPED_Lab.py"),
    ("🚦", "SRM Checker", "Detect Sample Ratio Mismatch — the silent experiment killer — before you trust any result.", "pages/6_SRM_Checker.py"),
    ("📚", "Pitfalls Playbook", "Simpson's Paradox, novelty effects, and other traps — with interactive demonstrations.", "pages/7_Pitfalls_Playbook.py"),
]

col_count = 3
rows = [cards[i : i + col_count] for i in range(0, len(cards), col_count)]

for row in rows:
    cols = st.columns(col_count)
    for col, (emoji, title, desc, target) in zip(cols, row):
        with col:
            st.markdown(
                f"""
                <div class="description-box" style="min-height: 165px;">
                    <div style="font-size: 1.8rem;">{emoji}</div>
                    <div style="font-weight: 700; font-size: 1.05rem; margin: 6px 0;">{title}</div>
                    <div class="explanation-text" style="font-size: 0.88rem;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.info("👈 Use the sidebar to open any lab.")

st.divider()
st.caption(
    "Built with Streamlit, SciPy, and Plotly · Frequentist and Bayesian inference, sequential testing, "
    "CUPED, and SRM detection implemented from first principles."
)
