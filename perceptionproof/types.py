"""Typed schemas shared across the harness and backends.

These are the only data contracts the open science code depends on. A backend
(local or Aweb/Maestro) must produce `SceneBundle` and `ModelOutput`; everything
downstream (signals, scoring) is backend-agnostic. See docs/ARCHITECTURE.md sec 2.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class _Arr(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class TrajectoryMode(_Arr):
    """One predicted future trajectory mode in the ego BEV frame.

    waypoints: (T, 2) float array. weight: mode probability in [0, 1].
    """

    waypoints: np.ndarray
    weight: float = 1.0


class OccupancyField(_Arr):
    """Per-voxel occupancy probability over a fixed grid, plus the ego-corridor mask.

    prob: (X, Y, Z) in [0, 1]. corridor_mask: bool array, the planned-path sweep
    and its occluded frustums (MATHEMATICS sec 2.3, set C).
    """

    prob: np.ndarray
    corridor_mask: np.ndarray


class ModelOutput(BaseModel):
    """One model's prediction for one segment."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_id: str
    weights_sha256: str
    trajectory_modes: list[TrajectoryMode] = Field(default_factory=list)
    occupancy: OccupancyField | None = None
    reasoning_rollouts: list[TrajectoryMode] = Field(default_factory=list)  # for S4 (VLA)
    free_space_confidence: float | None = None  # planner "corridor is clear" score, for S3 conflict
    latency_ms: float | None = None
    cost_usd: float | None = None


class SceneBundle(BaseModel):
    """Everything about one segment except model predictions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    segment_id: str
    dataset_version: str
    drive_id: str  # cluster-bootstrap / GroupKFold unit (MATHEMATICS sec 5)
    rfs: float | None = None  # human ground truth in [0, 10]; None at inference-only time
    # Multi-frame predictions for the temporal signal S2 are attached by the backend
    # as a mapping {scene_time_k: ego_pose}; kept opaque here.


class Receipt(BaseModel):
    """Tamper-evident provenance for one step. See docs/ARCHITECTURE.md sec 5."""

    run_id: str
    step: str
    segment_id: str | None = None
    inputs_hash: str
    outputs_hash: str
    prev_receipt_hash: str
    content_hash: str
    signature: str
    extra: dict = Field(default_factory=dict)
