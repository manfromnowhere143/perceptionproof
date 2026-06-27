"""Dataset adapters. The harness depends only on the DatasetAdapter interface, so the
same pipeline runs over a fixture, WOD-E2E, or NAVSIM by swapping the adapter.

- FixtureDatasetAdapter: deterministic in-memory data for testing the LocalBackend
  composition offline (no real data, no GPU).
- WodE2EAdapter / NavsimAdapter: real adapters, implemented at P2 once the licensed data
  is local. They are documented stubs here — no real frames are redistributed.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from .types import SceneBundle


def difficulty(text: str) -> float:
    """Deterministic latent in [0,1] from a string (fixtures plant difficulty -> RFS)."""
    return (int(hashlib.sha256(text.encode()).hexdigest(), 16) % 1000) / 999.0


class DatasetAdapter(Protocol):
    def segment_ids(self) -> list[str]: ...
    def load(self, segment_id: str) -> SceneBundle: ...


class FixtureDatasetAdapter:
    """In-memory deterministic dataset. Plants difficulty -> RFS (harder -> lower rating)
    so the production seam can be validated end-to-end without real data."""

    def __init__(self, segment_ids: list[str]) -> None:
        self._ids = list(segment_ids)

    def segment_ids(self) -> list[str]:
        return list(self._ids)

    def load(self, segment_id: str) -> SceneBundle:
        d = difficulty(segment_id)
        rfs = float(np.clip(10.0 * (1.0 - d), 0.0, 10.0))
        return SceneBundle(
            segment_id=segment_id,
            dataset_version="fixture-v0",
            drive_id=segment_id.rsplit("_", 1)[0],
            rfs=rfs,
        )


class WodE2EAdapter:
    """Real WOD-E2E adapter (Waymo Open Dataset license; data NOT redistributed).

    Loads 8-camera segments + ego state + high-level routing, and the per-trajectory
    Rater Feedback Score on the validation split. Implemented at P2 once the licensed
    tfrecords are local. Source + license: https://waymo.com/open
    """

    def __init__(self, data_root: str) -> None:
        self.data_root = data_root

    def segment_ids(self) -> list[str]:
        raise NotImplementedError("P2: enumerate WOD-E2E validation segments at data_root")

    def load(self, segment_id: str) -> SceneBundle:
        raise NotImplementedError("P2: decode cameras + ego + routing + RFS for the segment")


class NavsimAdapter:
    """Real NAVSIM adapter over OpenScene/nuPlan frames (navtest split; non-commercial
    research license). Yields scenes whose trajectories are scored by PDMS rather than RFS.
    Implemented at P2. Source: https://github.com/autonomousvision/navsim
    """

    def __init__(self, data_root: str) -> None:
        self.data_root = data_root

    def segment_ids(self) -> list[str]:
        raise NotImplementedError("P2: enumerate navtest tokens at data_root")

    def load(self, segment_id: str) -> SceneBundle:
        raise NotImplementedError("P2: load sensor blobs + ego state + route for the token")
