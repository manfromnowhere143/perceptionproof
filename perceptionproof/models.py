"""Model runners. Each runner wraps one model and returns a ModelOutput for a scene.
An ensemble of runners is what produces disagreement (S1). The harness depends only on
the ModelRunner interface, so real models and fixtures are interchangeable.

- FixtureModelRunner: deterministic, GPU-free; spread grows with planted scene difficulty
  so an ensemble yields disagreement correlated with (planted) low RFS — for testing the
  LocalBackend seam only.
- NavsimAgentRunner / E2EPlannerRunner / VlaRunner: real wrappers, implemented at P2 on
  the GPU box.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from .dataset import difficulty
from .types import ModelOutput, SceneBundle, TrajectoryMode


class ModelRunner(Protocol):
    model_id: str
    weights_sha256: str

    def predict(self, scene: SceneBundle) -> ModelOutput: ...


class FixtureModelRunner:
    """Deterministic fixture model. All runners share a per-scene base trajectory; each
    adds jitter whose scale grows with planted difficulty, so ensemble disagreement tracks
    (planted) low RFS. Not a real model — validates the LocalBackend composition offline.
    """

    def __init__(self, index: int, horizon: int = 6, seed: int = 20260627) -> None:
        self.index = index
        self.horizon = horizon
        self._seed = seed
        self.model_id = f"fixture-model-{index}"
        self.weights_sha256 = "fixture"

    def predict(self, scene: SceneBundle) -> ModelOutput:
        d = difficulty(scene.segment_id)
        mask = int(hashlib.sha256(scene.segment_id.encode()).hexdigest(), 16) & 0xFFFFFFFF
        base_rng = np.random.default_rng(self._seed ^ mask)  # shared across runners
        base = np.cumsum(base_rng.normal(size=(self.horizon, 2)) * 0.1, axis=0)
        jitter_rng = np.random.default_rng((self._seed ^ mask ^ (self.index + 1)) & 0xFFFFFFFF)
        jitter = jitter_rng.normal(scale=0.05 + 1.5 * d, size=(self.horizon, 2))
        return ModelOutput(
            model_id=self.model_id,
            weights_sha256=self.weights_sha256,
            trajectory_modes=[TrajectoryMode(waypoints=base + jitter, weight=1.0)],
        )


class NavsimAgentRunner:
    """Real runner wrapping a NAVSIM baseline agent (e.g., TransFuser, ego-status MLP) to
    emit a trajectory ModelOutput. Implemented at P2. https://github.com/autonomousvision/navsim
    """

    def __init__(self, agent_name: str, weights_sha256: str, checkpoint: str | None = None) -> None:
        self.agent_name = agent_name
        self.checkpoint = checkpoint
        self.model_id = f"navsim:{agent_name}"
        self.weights_sha256 = weights_sha256

    def predict(self, scene: SceneBundle) -> ModelOutput:
        raise NotImplementedError("P2: run NAVSIM agent inference on the GPU box")


class E2EPlannerRunner:
    """Real runner for a camera E2E planner producing a trajectory (and optionally an
    occupancy field for S3). Implemented at P2."""

    def __init__(self, model_id: str, weights_sha256: str) -> None:
        self.model_id = model_id
        self.weights_sha256 = weights_sha256

    def predict(self, scene: SceneBundle) -> ModelOutput:
        raise NotImplementedError("P2: run E2E planner inference")
