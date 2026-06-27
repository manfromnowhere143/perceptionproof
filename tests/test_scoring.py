"""Power and selective-prediction estimators, validated on known-answer inputs."""

from __future__ import annotations

import numpy as np

from perceptionproof.scoring import min_detectable_rho, risk_coverage


def test_min_detectable_rho_decreases_with_n():
    assert min_detectable_rho(30) > min_detectable_rho(300)
    r = min_detectable_rho(100)
    assert 0.0 < r < 1.0


def test_risk_coverage_perfect_signal_zero_eaurc():
    # signal equals loss ordering -> retained set is always the best possible -> E-AURC = 0
    loss = np.array([0.0, 0.0, 1.0, 1.0])
    signal = loss.copy()
    out = risk_coverage(signal, loss)
    assert abs(out["e_aurc"]) < 1e-12


def test_risk_coverage_worst_signal_has_positive_eaurc():
    loss = np.array([0.0, 0.0, 1.0, 1.0])
    signal = -loss  # most-confident-first picks the failures first: worst ordering
    out = risk_coverage(signal, loss)
    assert out["e_aurc"] > 0.0


def test_risk_coverage_full_coverage_risk_is_mean_loss():
    loss = np.array([0.0, 1.0, 1.0, 0.0])
    out = risk_coverage(np.arange(4.0), loss)
    assert abs(out["risks"][-1] - loss.mean()) < 1e-12
