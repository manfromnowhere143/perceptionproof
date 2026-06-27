"""Validity metrics and the statistical protocol. Implements docs/MATHEMATICS.md
sec 4-5. Backend-agnostic: consumes arrays of signal values and RFS labels.

Every estimator here is validated on synthetic data of known answer (tests/) BEFORE
it is ever run on a real RFS label — so that a number on real data reflects the world,
not a bug in the pipeline. This is the de-risking order, not an afterthought.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import average_precision_score, roc_auc_score

# np.trapz was renamed to np.trapezoid in NumPy 2.0; support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------------
# sec 5 -- power (deterministic; referenced by PREREGISTRATION)
# ---------------------------------------------------------------------------
def min_detectable_rho(n: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Fisher-z minimum detectable Spearman rho at sample size n (MATHEMATICS sec 5)."""
    if n <= 3:
        raise ValueError("need n > 3 for the Fisher-z approximation")
    z = norm.ppf(1.0 - alpha / 2.0) + norm.ppf(power)
    return float(np.tanh(z / np.sqrt(n - 3)))


# ---------------------------------------------------------------------------
# sec 4.1 -- predictive correlation (H1), cluster bootstrap + permutation null
# ---------------------------------------------------------------------------
def spearman_with_ci(
    signal,
    neg_rfs,
    drive_ids,
    n_boot: int = 10_000,
    seed: int = 20260627,
    n_perm: int | None = None,
) -> dict:
    """Spearman rho with drive-level cluster-bootstrap 95% CI and a permutation p-value.

    Resampling is at the drive level (not the segment level) so correlated segments
    from one drive cannot inflate confidence (MATHEMATICS sec 5).
    """
    rng = np.random.default_rng(seed)
    signal = np.asarray(signal, dtype=float)
    y = np.asarray(neg_rfs, dtype=float)
    drives = np.asarray(drive_ids)
    rho = float(spearmanr(signal, y).statistic)

    uniq = np.unique(drives)
    idx_by_drive = {d: np.where(drives == d)[0] for d in uniq}
    boots: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_drive[d] for d in sampled])
        if rows.size < 3:
            continue
        r = spearmanr(signal[rows], y[rows]).statistic
        if not np.isnan(r):
            boots.append(r)
    lo, hi = np.percentile(boots, [2.5, 97.5])

    n_perm = n_perm if n_perm is not None else n_boot
    perm = np.empty(n_perm)
    for i in range(n_perm):
        perm[i] = abs(spearmanr(signal, rng.permutation(y)).statistic)
    p = float((1 + np.sum(perm >= abs(rho))) / (n_perm + 1))

    return {"rho": rho, "ci_low": float(lo), "ci_high": float(hi), "p_value": p, "n": int(signal.size)}


# ---------------------------------------------------------------------------
# sec 4.2 -- failure mining (H3)
# ---------------------------------------------------------------------------
def failure_mining(signal, failure, ks: tuple[int, ...] = (5, 10)) -> dict:
    """AUROC, Average Precision, and precision@k vs base rate."""
    s = np.asarray(signal, dtype=float)
    y = np.asarray(failure).astype(int)
    has_both = np.unique(y).size > 1
    order = np.argsort(-s)  # highest signal first = most-suspected failures
    prec_at_k = {}
    for k in ks:
        kk = min(k, s.size)
        prec_at_k[str(k)] = float(y[order[:kk]].mean())
    return {
        "auroc": float(roc_auc_score(y, s)) if has_both else float("nan"),
        "ap": float(average_precision_score(y, s)) if has_both else float("nan"),
        "base_rate": float(y.mean()),
        "precision_at_k": prec_at_k,
    }


# ---------------------------------------------------------------------------
# sec 4.3 -- selective prediction (the deployable framing)
# ---------------------------------------------------------------------------
def risk_coverage(signal: np.ndarray, loss: np.ndarray) -> dict:
    """Risk-coverage curve, AURC and E-AURC (Geifman & El-Yaniv 2017; MATHEMATICS sec 4.3).

    Retain the lowest-`signal` fraction (most confident); risk = mean `loss` over the
    retained set. Lower AURC is better; E-AURC subtracts the oracle (order by true loss).
    """
    signal = np.asarray(signal, dtype=float)
    loss = np.asarray(loss, dtype=float)
    if signal.shape != loss.shape:
        raise ValueError("signal and loss must align")
    n = signal.shape[0]
    coverages = np.arange(1, n + 1) / n

    order = np.argsort(signal, kind="stable")
    risks = np.cumsum(loss[order]) / np.arange(1, n + 1)
    aurc = float(_trapz(risks, coverages))

    risks_oracle = np.cumsum(np.sort(loss)) / np.arange(1, n + 1)
    aurc_oracle = float(_trapz(risks_oracle, coverages))

    return {
        "coverages": coverages.tolist(),
        "risks": risks.tolist(),
        "aurc": aurc,
        "e_aurc": aurc - aurc_oracle,
    }


# ---------------------------------------------------------------------------
# sec 4.4 -- ranking-inversion recovery (H2, the centerpiece)
# ---------------------------------------------------------------------------
def _kendall_distance(scores_a, scores_b) -> int:
    """Number of discordant ordered pairs between two score vectors over the same items."""
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    m = a.size
    k = 0
    for i in range(m):
        for j in range(i + 1, m):
            sa = np.sign(a[i] - a[j])
            sb = np.sign(b[i] - b[j])
            if sa != 0 and sb != 0 and sa != sb:
                k += 1
    return int(k)


def kendall_inversion_recovery(
    open_loop_scores,
    signal_per_method,
    ground_truth_scores,
    lam: float,
    drive_ids=None,  # reserved for segment-level bootstrap CI once method aggregation is wired
    n_boot: int = 10_000,
) -> dict:
    """Does penalizing the open-loop score by the long-tail signal recover the true
    method ranking? H2 holds iff k_adj < k_open (MATHEMATICS sec 4.4).

    All inputs are per-method scalars. drive_ids/n_boot are accepted for interface
    stability; the bootstrap CI is added at P4 once segment->method aggregation is wired.
    """
    o = np.asarray(open_loop_scores, dtype=float)
    g = np.asarray(ground_truth_scores, dtype=float)
    s = np.asarray(signal_per_method, dtype=float)
    adjusted = o - lam * s
    k_open = _kendall_distance(o, g)
    k_adj = _kendall_distance(adjusted, g)
    return {"k_open": k_open, "k_adj": k_adj, "improvement": k_open - k_adj, "lambda": float(lam)}


# ---------------------------------------------------------------------------
# sec 4.5 -- information-theoretic effect size
# ---------------------------------------------------------------------------
def mutual_information(signal, failure, seed: int = 20260627, n_perm: int = 200) -> dict:
    """I(signal; failure) via a KNN (KSG-style) estimator, with a permutation null."""
    rng = np.random.default_rng(seed)
    x = np.asarray(signal, dtype=float).reshape(-1, 1)
    y = np.asarray(failure).astype(int)
    mi = float(mutual_info_classif(x, y, random_state=seed)[0])
    null = np.array(
        [float(mutual_info_classif(x, rng.permutation(y), random_state=seed)[0]) for _ in range(n_perm)]
    )
    p = float((1 + np.sum(null >= mi)) / (n_perm + 1))
    return {"mi": mi, "null_mean": float(null.mean()), "p_value": p}


# ---------------------------------------------------------------------------
# sec 5 -- multiple comparisons
# ---------------------------------------------------------------------------
def benjamini_hochberg(pvalues: dict[str, float], q: float = 0.05) -> dict[str, bool]:
    """Benjamini-Hochberg FDR control across the pre-registered test family."""
    names = list(pvalues.keys())
    p = np.array([pvalues[k] for k in names], dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresh = (np.arange(1, m + 1) / m) * q
    below = ranked <= thresh
    rejected_sorted = np.zeros(m, dtype=bool)
    if below.any():
        kmax = int(np.max(np.where(below)[0]))
        rejected_sorted[: kmax + 1] = True
    result: dict[str, bool] = {}
    for i, idx in enumerate(order):
        result[names[idx]] = bool(rejected_sorted[i])
    return result
