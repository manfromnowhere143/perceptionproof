"""S2 (temporal flicker), S3 (occupancy conflict), S4 (semantic entropy) on
hand-checkable inputs of known answer. Completes the signal layer on CPU."""

from __future__ import annotations

import numpy as np

from perceptionproof.signals import (
    s2_temporal_inconsistency,
    s3_occupancy_conflict,
    s4_semantic_entropy,
)
from perceptionproof.types import ModelOutput, OccupancyField, TrajectoryMode


def _traj_model(mid: str, waypoints) -> ModelOutput:
    return ModelOutput(
        model_id=mid,
        weights_sha256="x",
        trajectory_modes=[TrajectoryMode(waypoints=np.asarray(waypoints, dtype=float), weight=1.0)],
    )


# ---- S2: a temporally consistent model has zero flicker -------------------
def test_s2_consistent_forecast_zero_flicker():
    # ego advances +1 in x each tick (dtheta=0, dx=1, dy=0). A consistent model predicts
    # the same world line, so in each ego frame the forecast is [[1,0],[2,0],[3,0],[4,0]].
    fc = [[1, 0], [2, 0], [3, 0], [4, 0]]
    frames = [[_traj_model("a", fc)], [_traj_model("a", fc)]]
    ego = [(0.0, 1.0, 0.0)]
    assert s2_temporal_inconsistency(frames, ego) == 0.0


def test_s2_jumping_forecast_positive_and_monotone():
    fc = [[1, 0], [2, 0], [3, 0], [4, 0]]
    near = [[1, 0.3], [2, 0.3], [3, 0.3], [4, 0.3]]
    far = [[1, 3.0], [2, 3.0], [3, 3.0], [4, 3.0]]
    ego = [(0.0, 1.0, 0.0)]
    d_near = s2_temporal_inconsistency([[_traj_model("a", fc)], [_traj_model("a", near)]], ego)
    d_far = s2_temporal_inconsistency([[_traj_model("a", fc)], [_traj_model("a", far)]], ego)
    assert d_near > 0.0
    assert d_far > d_near


def test_s2_single_frame_is_zero():
    assert s2_temporal_inconsistency([[_traj_model("a", [[0, 0], [1, 0]])]], []) == 0.0


# ---- S3: occupancy conflict ------------------------------------------------
def _occ_model(prob, mask, free_conf=None):
    return ModelOutput(
        model_id="occ",
        weights_sha256="x",
        occupancy=OccupancyField(prob=np.asarray(prob, dtype=float), corridor_mask=np.asarray(mask, dtype=bool)),
        free_space_confidence=free_conf,
    )


def test_s3_confidently_empty_corridor_is_zero():
    prob = np.zeros((4, 1))
    mask = np.ones((4, 1), dtype=bool)
    # entropy of p=0 is 0 modulo the 1e-12 numerical-safety clip -> negligibly small
    assert s3_occupancy_conflict([_occ_model(prob, mask, free_conf=1.0)], theta_occ=0.5) < 1e-9


def test_s3_planner_clear_over_occupied_corridor_raises_signal():
    prob = np.full((4, 1), 0.9)  # corridor looks occupied
    mask = np.ones((4, 1), dtype=bool)
    pretending_safe = s3_occupancy_conflict([_occ_model(prob, mask, free_conf=1.0)], theta_occ=0.5)
    planner_cautious = s3_occupancy_conflict([_occ_model(prob, mask, free_conf=0.0)], theta_occ=0.5)
    # same scene; the conflict term fires only when the planner wrongly thinks it is clear
    assert pretending_safe > planner_cautious


# ---- S4: semantic entropy --------------------------------------------------
def _rollout_model(trajs):
    return ModelOutput(
        model_id="vla",
        weights_sha256="x",
        reasoning_rollouts=[TrajectoryMode(waypoints=np.asarray(t, dtype=float), weight=1.0) for t in trajs],
    )


def test_s4_agreeing_rollouts_zero_entropy():
    same = [[0, 0], [1, 0], [2, 0]]
    out = _rollout_model([same, same, same, same])
    assert s4_semantic_entropy([out], cluster_eps=0.5) == 0.0


def test_s4_two_equal_clusters_is_ln2():
    left = [[0, 0], [1, 0], [2, 0]]
    right = [[0, 0], [1, 5], [2, 10]]  # far apart -> distinct maneuver
    out = _rollout_model([left, right, left, right])
    assert abs(s4_semantic_entropy([out], cluster_eps=0.5) - np.log(2)) < 1e-9


def test_s4_three_equal_clusters_is_ln3():
    a = [[0, 0], [0, 0]]
    b = [[10, 0], [10, 0]]
    c = [[0, 10], [0, 10]]
    out = _rollout_model([a, b, c, a, b, c])
    assert abs(s4_semantic_entropy([out], cluster_eps=0.5) - np.log(3)) < 1e-9
