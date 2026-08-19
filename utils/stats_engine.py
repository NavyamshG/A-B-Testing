"""
Core statistical engine for CTR Inference Lab / A-B Testing Studio.
All math lives here so every page (frequentist, bayesian, sequential,
CUPED, SRM, sample size) shares one tested source of truth.
"""

import math
import numpy as np
import scipy.stats as stats


# ------------------------------------------------------------------
# Basic validation
# ------------------------------------------------------------------
def validate_counts(x: int, n: int, label: str):
    """Returns an error string if invalid, else None."""
    if n <= 0:
        return f"{label}: trials must be > 0."
    if x < 0:
        return f"{label}: clicks must be ≥ 0."
    if x > n:
        return f"{label}: clicks must be ≤ trials."
    return None


# ------------------------------------------------------------------
# Frequentist tests
# ------------------------------------------------------------------
def two_prop_ztest(x1, n1, x2, n2):
    p1_, p2_ = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return np.nan, np.nan
    z = (p2_ - p1_) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


def welch_ttest_bernoulli(x1, n1, x2, n2):
    p1_, p2_ = x1 / n1, x2 / n2
    if n1 <= 1 or n2 <= 1:
        return np.nan, np.nan, np.nan
    s1_sq = (n1 / (n1 - 1)) * p1_ * (1 - p1_)
    s2_sq = (n2 / (n2 - 1)) * p2_ * (1 - p2_)
    se = math.sqrt(s1_sq / n1 + s2_sq / n2)
    if se == 0:
        return np.nan, np.nan, np.nan
    t = (p2_ - p1_) / se
    num = (s1_sq / n1 + s2_sq / n2) ** 2
    den = ((s1_sq / n1) ** 2) / (n1 - 1) + ((s2_sq / n2) ** 2) / (n2 - 1)
    df = num / den if den > 0 else np.nan
    p = 2 * (1 - stats.t.cdf(abs(t), df))
    return t, df, p


def fishers_exact(x1, n1, x2, n2):
    table = np.array([[x1, n1 - x1], [x2, n2 - x2]])
    oddsratio, p = stats.fisher_exact(table, alternative="two-sided")
    return oddsratio, p


def wald_ci_diff(x1, n1, x2, n2, alpha=0.05):
    p1_, p2_ = x1 / n1, x2 / n2
    d = p2_ - p1_
    se = math.sqrt(p1_ * (1 - p1_) / n1 + p2_ * (1 - p2_) / n2)
    z = stats.norm.ppf(1 - alpha / 2)
    return d, d - z * se, d + z * se


def wilson_ci_single(x, n, alpha=0.05):
    if n == 0:
        return np.nan, np.nan
    z = stats.norm.ppf(1 - alpha / 2)
    p = x / n
    denom = 1 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1 - p) / n) + (z**2) / (4 * n**2))
    return max(0.0, center - half), min(1.0, center + half)


def newcombe_ci_diff(x1, n1, x2, n2, alpha=0.05):
    l1, u1 = wilson_ci_single(x1, n1, alpha)
    l2, u2 = wilson_ci_single(x2, n2, alpha)
    d = (x2 / n2) - (x1 / n1)
    return d, (l2 - u1), (u2 - l1)


# ------------------------------------------------------------------
# Bayesian
# ------------------------------------------------------------------
def beta_credible_interval(samples, level=0.95):
    lo = (1 - level) / 2
    hi = 1 - lo
    return np.percentile(samples, [100 * lo, 100 * hi])


def expected_loss(control_samples, variant_samples):
    """Expected loss of choosing Variant when Control is actually better, and vice versa."""
    loss_choose_variant = np.mean(np.maximum(control_samples - variant_samples, 0))
    loss_choose_control = np.mean(np.maximum(variant_samples - control_samples, 0))
    return loss_choose_variant, loss_choose_control


# ------------------------------------------------------------------
# Sample size / power
# ------------------------------------------------------------------
def sample_size_two_proportions(p1, mde, alpha=0.05, power=0.8, two_sided=True):
    """Classic normal-approximation sample size per group for a two-proportion test."""
    p2 = p1 + mde
    p2 = min(max(p2, 1e-9), 1 - 1e-9)
    z_alpha = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    p_bar = (p1 + p2) / 2
    numerator = (
        z_alpha * math.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    denominator = (p2 - p1) ** 2
    if denominator == 0:
        return np.inf
    return math.ceil(numerator / denominator)


def power_for_n(p1, mde, n_per_group, alpha=0.05, two_sided=True):
    """Given fixed n per group, compute achieved power."""
    p2 = p1 + mde
    p2 = min(max(p2, 1e-9), 1 - 1e-9)
    z_alpha = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    se0 = math.sqrt(2 * ((p1 + p2) / 2) * (1 - (p1 + p2) / 2) / n_per_group)
    se1 = math.sqrt(p1 * (1 - p1) / n_per_group + p2 * (1 - p2) / n_per_group)
    if se1 == 0:
        return np.nan
    z = (abs(p2 - p1) - z_alpha * se0) / se1
    return float(stats.norm.cdf(z))


# ------------------------------------------------------------------
# Sequential testing / always-valid inference
# ------------------------------------------------------------------
def obrien_fleming_boundary(alpha, k, total_looks):
    """
    Approximate O'Brien-Fleming alpha-spending boundary (z-scale) for look k of
    total_looks. Conservative early, relaxes toward the nominal alpha at the final look.
    """
    if k <= 0:
        k = 1
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    # classic O'Brien-Fleming scaling: boundary shrinks as sqrt(total_looks / k)
    boundary_z = z_alpha * math.sqrt(total_looks / k)
    return boundary_z


def mixture_sequential_p_value(z_stat, n_effective, tau=1.0):
    """
    A simple 'always-valid' p-value approximation using a mixture (mSPRT-style)
    sequential test with a normal mixing distribution of variance tau^2.
    Valid to peek at continuously without alpha inflation (approximately).
    """
    if not np.isfinite(z_stat) or n_effective <= 0:
        return np.nan
    # Likelihood ratio for a normal mixture prior N(0, tau^2) on the effect,
    # evaluated at the observed z statistic scaled by effective sample size.
    lr = math.sqrt(1 / (1 + n_effective * tau**2)) * math.exp(
        (n_effective * tau**2 * z_stat**2) / (2 * (1 + n_effective * tau**2))
    )
    # Convert likelihood ratio to an always-valid p-value bound: p <= 1/LR
    p_bound = min(1.0, 1 / lr) if lr > 0 else 1.0
    return p_bound


# ------------------------------------------------------------------
# CUPED (Controlled-experiment Using Pre-Experiment Data)
# ------------------------------------------------------------------
def cuped_adjust(y, x, theta=None):
    """
    y: post-experiment metric (array-like)
    x: pre-experiment covariate for the same units (array-like)
    Returns (y_adjusted, theta_used, variance_reduction_pct)
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if theta is None:
        cov_xy = np.cov(x, y, ddof=1)[0, 1]
        var_x = np.var(x, ddof=1)
        theta = cov_xy / var_x if var_x > 0 else 0.0
    y_adj = y - theta * (x - np.mean(x))
    var_before = np.var(y, ddof=1)
    var_after = np.var(y_adj, ddof=1)
    reduction = 100 * (1 - var_after / var_before) if var_before > 0 else 0.0
    return y_adj, theta, reduction


# ------------------------------------------------------------------
# Sample Ratio Mismatch (SRM)
# ------------------------------------------------------------------
def srm_chi_square(n1, n2, expected_ratio=0.5):
    """Chi-square goodness-of-fit test for sample ratio mismatch."""
    total = n1 + n2
    expected_n1 = total * expected_ratio
    expected_n2 = total * (1 - expected_ratio)
    chi2 = ((n1 - expected_n1) ** 2) / expected_n1 + ((n2 - expected_n2) ** 2) / expected_n2
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    return chi2, p_value


# ------------------------------------------------------------------
# Multiple testing correction
# ------------------------------------------------------------------
def bonferroni_correction(p_values, alpha=0.05):
    m = len(p_values)
    adj_alpha = alpha / m if m > 0 else alpha
    return adj_alpha, [p <= adj_alpha for p in p_values]


def benjamini_hochberg(p_values, alpha=0.05):
    """Returns list of booleans indicating which hypotheses are rejected under BH/FDR control."""
    p_arr = np.asarray(p_values)
    m = len(p_arr)
    order = np.argsort(p_arr)
    ranked = p_arr[order]
    thresholds = (np.arange(1, m + 1) / m) * alpha
    below = ranked <= thresholds
    if not np.any(below):
        return [False] * m
    max_k = np.max(np.where(below)[0])
    reject_sorted = np.zeros(m, dtype=bool)
    reject_sorted[: max_k + 1] = True
    reject = np.zeros(m, dtype=bool)
    reject[order] = reject_sorted
    return reject.tolist()


def pct(x):
    return f"{x:.2%}" if np.isfinite(x) else "NA"
