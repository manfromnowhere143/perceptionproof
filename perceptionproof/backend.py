"""The PerceptionBackend interface (docs/ARCHITECTURE.md sec 2) and the open,
deterministic LocalBackend. The Aweb/Maestro backend lives in the proprietary
aweb repo and implements the same Protocol.

The open science (signals, scoring) depends ONLY on this interface, so an outside
engineer reproduces every figure with LocalBackend and zero Aweb access.
"""

from __future__ import annotations

from typing import Protocol

from .types import ModelOutput, Receipt, SceneBundle


class ModelLock:
    """Loaded protocol/models.lock.json — frozen model ids + weight hashes."""


class StepRecord:
    """The payload hashed + signed into a Receipt (run_id, step, inputs, outputs)."""


class PerceptionBackend(Protocol):
    def ingest(self, segment_id: str) -> SceneBundle: ...
    def run_models(self, scene: SceneBundle, lock: ModelLock) -> list[ModelOutput]: ...
    def emit_receipt(self, record: StepRecord) -> Receipt: ...


class LocalBackend:
    """Deterministic, dependency-light backend for full open reproducibility.

    ingest:      loads a SceneBundle from a user-provided local dataset path (no
                 redistribution; user accepts dataset license themselves).
    run_models:  runs the pinned open models, OR replays cached ModelOutput fixtures
                 so results reproduce byte-for-byte without a GPU.
    emit_receipt: hash-chains (blake3 run_id, sha256 content) and signs (ed25519)
                 with a repo-local dev key; chain + signatures publicly verifiable.

    P1 status: interface fixed. ingest/run_models implemented at P2, receipt signing
    implemented alongside (it is small and self-contained).
    """

    def __init__(self, run_id: str, signing_key_path: str) -> None:
        self._run_id = run_id
        self._signing_key_path = signing_key_path
        self._prev_hash = "0" * 64  # genesis

    def ingest(self, segment_id: str) -> SceneBundle:
        raise NotImplementedError("P2")

    def run_models(self, scene: SceneBundle, lock: ModelLock) -> list[ModelOutput]:
        raise NotImplementedError("P2")

    def emit_receipt(self, record: StepRecord) -> Receipt:
        raise NotImplementedError("P2")
