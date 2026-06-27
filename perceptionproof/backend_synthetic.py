"""SyntheticBackend — a deterministic, GPU-free backend that PROVES THE PIPELINE
PLUMBING ONLY. It plants a hidden 'difficulty' latent per segment that drives both
(a) the human RFS (higher difficulty -> lower rating) and (b) model trajectory
disagreement (higher difficulty -> more spread). This lets the end-to-end machine
(ingest -> models -> signals -> score -> receipts) be verified on data of KNOWN
answer.

It is NOT a scientific result. The correlation it produces is planted by construction,
not discovered. Real findings require real models on real WOD-E2E frames (P2+).
"""

from __future__ import annotations

import hashlib

import numpy as np

from .receipts import GENESIS_HASH, ReceiptSigner, canonical_json, sha256_hex
from .types import ModelOutput, SceneBundle, TrajectoryMode


def _u01(text: str) -> float:
    """Deterministic uniform-ish value in [0,1] from a string."""
    return (int(hashlib.sha256(text.encode()).hexdigest(), 16) % 1000) / 999.0


def hash_model_outputs(outputs: list[ModelOutput]) -> str:
    """Deterministic content hash over a list of model outputs (for receipts)."""
    parts = [
        {
            "model_id": o.model_id,
            "weights": o.weights_sha256,
            "modes": [sha256_hex(np.ascontiguousarray(m.waypoints).tobytes()) for m in o.trajectory_modes],
        }
        for o in outputs
    ]
    return sha256_hex(canonical_json(parts))


class SyntheticBackend:
    def __init__(self, run_id: str, signer: ReceiptSigner, n_models: int = 3,
                 horizon: int = 6, seed: int = 20260627) -> None:
        self.run_id = run_id
        self.signer = signer
        self.n_models = n_models
        self.horizon = horizon
        self.seed = seed
        self._prev = GENESIS_HASH

    def ingest(self, segment_id: str) -> SceneBundle:
        d = _u01(segment_id)  # difficulty in [0,1]
        rfs = float(np.clip(10.0 * (1.0 - d), 0.0, 10.0))  # harder -> lower human rating
        drive = segment_id.rsplit("_", 1)[0]
        return SceneBundle(segment_id=segment_id, dataset_version="synthetic-v0", drive_id=drive, rfs=rfs)

    def run_models(self, scene: SceneBundle, lock=None) -> list[ModelOutput]:
        d = _u01(scene.segment_id)
        mask = int(hashlib.sha256(scene.segment_id.encode()).hexdigest(), 16) & 0xFFFFFFFF
        rng = np.random.default_rng(self.seed ^ mask)
        base = np.cumsum(rng.normal(size=(self.horizon, 2)) * 0.1, axis=0)
        outputs: list[ModelOutput] = []
        for m in range(self.n_models):
            jitter = rng.normal(scale=0.05 + 1.5 * d, size=(self.horizon, 2))  # spread grows with difficulty
            outputs.append(
                ModelOutput(
                    model_id=f"synthetic-model-{m}",
                    weights_sha256="synthetic",
                    trajectory_modes=[TrajectoryMode(waypoints=base + jitter, weight=1.0)],
                )
            )
        return outputs

    def emit_receipt(self, *, step: str, segment_id: str | None, inputs_hash: str,
                     outputs_hash: str, extra: dict | None = None) -> dict:
        rec = self.signer.sign_step(
            run_id=self.run_id,
            step=step,
            segment_id=segment_id,
            inputs_hash=inputs_hash,
            outputs_hash=outputs_hash,
            prev_receipt_hash=self._prev,
            extra=extra or {},
        )
        self._prev = rec["content_hash"]
        return rec
