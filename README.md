# 🧪 A/B Testing Studio

An interactive, multi-page Streamlit application for designing, running, and correctly
interpreting A/B tests — built to demonstrate both applied statistics/experimentation depth
and software engineering craft.

**Live demo:** _(add your Streamlit Community Cloud link here)_

## What it does

| Page | What it covers |
|---|---|
| 🏠 Home | Overview and navigation hub |
| 📐 Sample Size Calculator | Required sample size, power curves, MDE trade-offs |
| 📊 Frequentist Lab | Z-test vs. Welch's T-test vs. Fisher's Exact on the same data, Wald & Newcombe CIs |
| 🧠 Bayesian Lab | Beta-Binomial posteriors, credible intervals, P(Variant wins), expected loss |
| ⏱️ Sequential Testing | Simulates the false-positive inflation from "peeking," and an always-valid (mSPRT-style) alternative |
| 🎛️ CUPED Lab | Variance reduction using pre-experiment covariates, with effective sample size savings |
| 🚦 SRM Checker | Chi-square test for Sample Ratio Mismatch, with a sensitivity curve |
| 📚 Pitfalls Playbook | Interactive Simpson's Paradox builder, multiple-testing correction demo (Bonferroni vs. BH), plus a field guide to novelty effects, interference, and metric mismatch |

## Why these features

Most portfolio A/B testing apps stop at "run a t-test." This one is built around the mistakes
that actually break experiments in production: peeking, sample ratio mismatch, underpowered
tests, and multiple comparisons — each with a working simulation, not just a definition.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
ab-studio/
├── app.py                  # Home page
├── pages/                  # One file per lab (Streamlit multi-page app)
├── utils/
│   ├── stats_engine.py     # All statistical logic — single source of truth
│   └── theme.py            # Shared design system (CSS, header components)
└── requirements.txt
```

## Tech

Streamlit · NumPy · SciPy · Plotly · Pandas — no black-box stats libraries; every test,
interval, and correction is implemented from first principles in `utils/stats_engine.py`.
