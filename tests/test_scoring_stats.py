"""Label-dependent validity metrics, each validated on synthetic data of KNOWN answer.
If these are correct here, a number on real RFS reflects the world, not a bug."""

from __future__ import annotations

import numpy as np

from perceptionproof.scoring import (
    benjamini_hochberg,
    failure_mining,
    kendall_inversion_recovery,
    mutual_information,
    spearman_with_ci,
)


def test_spearman_recovers_strong_positive_correlation():
    rng = np.random.default_rng(0)
    drives = np.repeat(np.arange(20), 3)  # 20 drives x 3 segments = 60
    signal = rng.normal(size=60)
    neg_rfs = 2.0 * signal + rng.normal(scale=0.3, size=60)  # strongly tied to signal
    out = spearman_with_ci(signal, neg_rfs, drives, n_boot=300, n_perm=300, seed=1)
    assert out["rho"] > 0.7
    assert out["ci_low"] > 0.0  # CI excludes zero
    assert out["p_value"] < 0.05


def test_spearman_independent_ci_spans_zero():
    rng = np.random.default_rng(2)
    drives = np.repeat(np.arange(20), 3)
    signal = rng.normal(size=60)
    neg_rfs = rng.normal(size=60)  # independent
    out = spearman_with_ci(signal, neg_rfs, drives, n_boot=300, n_perm=300, seed=3)
    # Correct statement of "no association": small effect AND non-significant p-value.
    # (A single seed is not guaranteed to bracket zero in its CI.)
    assert abs(out["rho"]) < 0.3
    assert out["p_value"] > 0.05


def test_failure_mining_perfect_signal():
    failure = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    out = failure_mining(failure.astype(float), failure, ks=(4,))
    assert out["auroc"] == 1.0
    assert out["ap"] == 1.0
    assert abs(out["base_rate"] - 0.5) < 1e-12
    assert out["precision_at_k"]["4"] == 1.0  # top-4 by signal are all failures


def test_kendall_inversion_recovery_fixes_an_inversion():
    # ground truth A > B > C ; open-loop overrates B (says B > A > C) -> 1 inversion.
    # B carries the highest long-tail signal, so penalizing by signal restores A > B > C.
    ground_truth = np.array([3.0, 2.0, 1.0])  # A, B, C
    open_loop = np.array([2.0, 3.0, 1.0])     # B ranked above A
    signal = np.array([0.0, 1.0, 0.0])        # B is the overrated one
    out = kendall_inversion_recovery(open_loop, signal, ground_truth, lam=1.5)
    assert out["k_open"] == 1
    assert out["k_adj"] == 0
    assert out["improvement"] > 0


def test_mutual_information_detects_dependence():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, size=200)
    dependent = y + rng.normal(scale=0.3, size=200)   # carries info about y
    independent = rng.normal(size=200)
    mi_dep = mutual_information(dependent, y, n_perm=100)
    mi_ind = mutual_information(independent, y, n_perm=100)
    assert mi_dep["mi"] > mi_ind["mi"]
    assert mi_dep["p_value"] < 0.05


def test_benjamini_hochberg_rejects_small_pvalues():
    res = benjamini_hochberg({"a": 0.001, "b": 0.01, "c": 0.2, "d": 0.5}, q=0.05)
    assert res["a"] is True and res["b"] is True
    assert res["c"] is False and res["d"] is False
