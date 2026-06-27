"""S1 disagreement + trajectory distance, on hand-checkable inputs."""

from __future__ import annotations

import numpy as np

from perceptionproof.signals import s1_ensemble_disagreement, trajectory_distance
from perceptionproof.types import ModelOutput, TrajectoryMode


def _model(mid: str, waypoints) -> ModelOutput:
    return ModelOutput(
        model_id=mid,
        weights_sha256="deadbeef",
        trajectory_modes=[TrajectoryMode(waypoints=np.asarray(waypoints, dtype=float), weight=1.0)],
    )


def test_trajectory_distance_identity_is_zero():
    a = [[0, 0], [1, 0], [2, 0]]
    assert trajectory_distance(np.array(a, float), np.array(a, float)) == 0.0


def test_trajectory_distance_known_value():
    a = np.array([[0, 0], [0, 0]], float)
    b = np.array([[3, 4], [0, 0]], float)  # first waypoint L2 = 5, second = 0; mean = 2.5
    assert trajectory_distance(a, b) == 2.5


def test_s1_identical_models_zero_disagreement():
    traj = [[0, 0], [1, 1], [2, 2]]
    out = [_model("a", traj), _model("b", traj)]
    assert s1_ensemble_disagreement(out, sigma=1.0) == 0.0


def test_s1_divergent_models_positive_and_monotone():
    base = [[0, 0], [1, 0], [2, 0]]
    near = [[0, 0], [1, 0.2], [2, 0.2]]
    far = [[0, 0], [1, 3.0], [2, 6.0]]
    d_near = s1_ensemble_disagreement([_model("a", base), _model("b", near)], sigma=1.0)
    d_far = s1_ensemble_disagreement([_model("a", base), _model("b", far)], sigma=1.0)
    assert d_near > 0.0
    assert d_far > d_near  # more divergence -> more disagreement


def test_s1_single_model_is_zero():
    assert s1_ensemble_disagreement([_model("a", [[0, 0], [1, 1]])], sigma=1.0) == 0.0
