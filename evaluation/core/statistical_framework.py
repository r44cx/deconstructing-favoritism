import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import pandas as pd
from typing import Tuple, List, Optional, Callable, Union
import warnings

from .config import (
    BOOTSTRAP_N, PERMUTATION_N, ALPHA, CONFIDENCE_LEVEL, 
    RANDOM_SEED, SIG_LEVELS
)

np.random.seed(RANDOM_SEED)

def bootstrap_ci(
    data: np.ndarray,
    metric_func: Callable = np.mean,
    n_bootstrap: int = BOOTSTRAP_N,
    ci: int = CONFIDENCE_LEVEL,
    seed: Optional[int] = None
) -> Tuple[float, float, float]:
    data = np.array(data)
    if len(data) == 0:
        return np.nan, np.nan, np.nan
    
    if seed is not None:
        np.random.seed(seed)
    
    point_estimate = metric_func(data)
    
    bootstrap_samples = np.random.choice(data, size=(n_bootstrap, len(data)), replace=True)
    bootstrap_stats = np.apply_along_axis(metric_func, 1, bootstrap_samples)
    
    lower_pct = (100 - ci) / 2
    upper_pct = 100 - lower_pct
    
    lower = np.percentile(bootstrap_stats, lower_pct)
    upper = np.percentile(bootstrap_stats, upper_pct)
    
    return float(point_estimate), float(lower), float(upper)


def bootstrap_ci_delta(
    group1: np.ndarray,
    group2: np.ndarray,
    n_bootstrap: int = BOOTSTRAP_N,
    ci: int = CONFIDENCE_LEVEL
) -> Tuple[float, float, float]:
    group1, group2 = np.array(group1), np.array(group2)
    n1, n2 = len(group1), len(group2)
    
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan, np.nan
    
    delta = np.mean(group1) - np.mean(group2)
    
    deltas = []
    for _ in range(n_bootstrap):
        boot1 = np.random.choice(group1, size=n1, replace=True)
        boot2 = np.random.choice(group2, size=n2, replace=True)
        deltas.append(np.mean(boot1) - np.mean(boot2))
    
    lower_pct = (100 - ci) / 2
    upper_pct = 100 - lower_pct
    
    lower = np.percentile(deltas, lower_pct)
    upper = np.percentile(deltas, upper_pct)
    
    return float(delta), float(lower), float(upper)


def permutation_test(
    group1: np.ndarray,
    group2: np.ndarray,
    metric_func: Callable = np.mean,
    n_permutations: int = PERMUTATION_N,
    alternative: str = 'two-sided'
) -> Tuple[float, float]:
    group1, group2 = np.array(group1), np.array(group2)
    observed_diff = metric_func(group1) - metric_func(group2)
    
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    
    perm_diffs = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_diff = metric_func(combined[:n1]) - metric_func(combined[n1:])
        perm_diffs.append(perm_diff)
    
    perm_diffs = np.array(perm_diffs)
    
    if alternative == 'two-sided':
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    elif alternative == 'greater':
        p_value = np.mean(perm_diffs >= observed_diff)
    elif alternative == 'less':
        p_value = np.mean(perm_diffs <= observed_diff)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")
    
    return float(observed_diff), float(p_value)


def compute_confusion_matrix_3x3(
    human_ratings: np.ndarray,
    judge_ratings: np.ndarray
) -> np.ndarray:
    human_ratings = np.array(human_ratings)
    judge_ratings = np.array(judge_ratings)
    
    def to_idx(r):
        return {1: 0, 0: 1, -1: 2}.get(r, 1)
    
    cm = np.zeros((3, 3), dtype=int)
    for h, j in zip(human_ratings, judge_ratings):
        if h in [1, 0, -1] and j in [1, 0, -1]:
            cm[to_idx(h), to_idx(j)] += 1
    
    return cm


def compute_faviscore(confusion_matrix: np.ndarray) -> float:
    W = np.array([
        [0, -1, -2],
        [1,  0, -1],
        [2,  1,  0]
    ])
    
    total_errors = np.sum(confusion_matrix) - np.trace(confusion_matrix)
    
    if total_errors == 0:
        return 0.0
    
    return float(np.sum(W * confusion_matrix) / total_errors)


def faviscore_with_ci(
    human_ratings: np.ndarray,
    judge_ratings: np.ndarray,
    n_bootstrap: int = BOOTSTRAP_N,
    ci: int = CONFIDENCE_LEVEL
) -> Tuple[float, float, float, List[float]]:
    human_ratings = np.array(human_ratings)
    judge_ratings = np.array(judge_ratings)
    
    valid_mask = np.isin(human_ratings, [1, 0, -1]) & np.isin(judge_ratings, [1, 0, -1])
    human_ratings = human_ratings[valid_mask]
    judge_ratings = judge_ratings[valid_mask]
    
    n = len(human_ratings)
    if n < 2:
        return np.nan, np.nan, np.nan, []
    
    cm = compute_confusion_matrix_3x3(human_ratings, judge_ratings)
    point_estimate = compute_faviscore(cm)
    
    bootstrap_faviscores = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        boot_h = human_ratings[indices]
        boot_j = judge_ratings[indices]
        boot_cm = compute_confusion_matrix_3x3(boot_h, boot_j)
        bootstrap_faviscores.append(compute_faviscore(boot_cm))
    
    lower_pct = (100 - ci) / 2
    upper_pct = 100 - lower_pct
    
    ci_lower = np.percentile(bootstrap_faviscores, lower_pct)
    ci_upper = np.percentile(bootstrap_faviscores, upper_pct)
    
    return float(point_estimate), float(ci_lower), float(ci_upper), bootstrap_faviscores


def faviscore_significant(ci_lower: float, ci_upper: float) -> bool:
    return ci_lower > 0 or ci_upper < 0


def chi_square_test(
    observed: np.ndarray,
    expected: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    observed = np.array(observed)
    
    if expected is None:
        expected = np.ones_like(observed) * np.sum(observed) / len(observed)
    else:
        expected = np.array(expected)
    
    chi2, p = stats.chisquare(observed, expected)
    
    return float(chi2), float(p)


def chi_square_contingency(
    contingency_table: np.ndarray
) -> Tuple[float, float, int, np.ndarray]:
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    return float(chi2), float(p), int(dof), expected


def significance_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return ""


def format_ci(
    point: float,
    lower: float,
    upper: float,
    as_percent: bool = True,
    decimals: int = 1
) -> str:
    if np.isnan(point):
        return "N/A"
    
    if as_percent:
        fmt = f"{{:.{decimals}f}}%"
        return f"{fmt.format(point * 100)} [{fmt.format(lower * 100)}, {fmt.format(upper * 100)}]"
    else:
        fmt = f"{{:.{decimals}f}}"
        return f"{fmt.format(point)} [{fmt.format(lower)}, {fmt.format(upper)}]"


def format_p_value(p: float, include_stars: bool = True) -> str:
    stars = significance_stars(p) if include_stars else ""
    
    if p < 0.001:
        return f"p < 0.001{stars}"
    elif p < 0.01:
        return f"p < 0.01{stars}"
    elif p < 0.05:
        return f"p = {p:.3f}{stars}"
    else:
        return f"p = {p:.2f}"


def format_latex_row(
    values: List[str],
    bold_first: bool = True
) -> str:
    if bold_first and values:
        values = [f"\\textbf{{{values[0]}}}"] + values[1:]
    
    return " & ".join(values) + " \\\\"


def full_comparison(
    group1: np.ndarray,
    group2: np.ndarray,
    group1_name: str = "Group 1",
    group2_name: str = "Group 2"
) -> dict:
    group1, group2 = np.array(group1), np.array(group2)
    
    m1, l1, u1 = bootstrap_ci(group1)
    
    m2, l2, u2 = bootstrap_ci(group2)
    
    delta, delta_low, delta_high = bootstrap_ci_delta(group1, group2)
    
    _, p_value = permutation_test(group1, group2)
    
    return {
        group1_name: {'mean': m1, 'ci_lower': l1, 'ci_upper': u1, 'n': len(group1)},
        group2_name: {'mean': m2, 'ci_lower': l2, 'ci_upper': u2, 'n': len(group2)},
        'delta': {'value': delta, 'ci_lower': delta_low, 'ci_upper': delta_high},
        'p_value': p_value,
        'significant': p_value < ALPHA,
        'stars': significance_stars(p_value)
    }
