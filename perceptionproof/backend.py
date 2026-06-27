"""The PerceptionBackend interface and the open, deterministic LocalBackend.

LocalBackend is the production-shaped backend: it composes a DatasetAdapter + a list of
ModelRunners + a ReceiptSigner. Its wiring is identical to the synthetic path; only the
adapter and runners differ. With real adapters/runners it runs the real study; with
fixtures it is fully tested offline (no data, no GPU). The Aweb/Maestro backend
implements the same interface in the proprietary repo (docs/ARCHITECTURE.md sec 2).
"""

from __future__ import annotations

from typing import Protocol

from .dataset import DatasetAdapter
from .models import ModelRunner
from .receipts import GENESIS_HASH, ReceiptSigner
from .types import ModelOutput, SceneBundle


class PerceptionBackend(Protocol):
    def ingest(self, segment_id: str) -> SceneBundle: ...
    def run_models(self, scene: SceneBundle) -> list[ModelOutput]: ...
    def emit_receipt(self, *, step: str, segment_id: str | None, inputs_hash: str,
                     outputs_hash: str, extra: dict | None = None) -> dict: ...


class LocalBackend:
    def __init__(self, run_id: str, adapter: DatasetAdapter,
                 runners: list[ModelRunner], signer: ReceiptSigner) -> None:
        self.run_id = run_id
        self.adapter = adapter
        self.runners = list(runners)
        self.signer = signer
        self._prev = GENESIS_HASH

    def ingest(self, segment_id: str) -> SceneBundle:
        return self.adapter.load(segment_id)

    def run_models(self, scene: SceneBundle) -> list[ModelOutput]:
        return [r.predict(scene) for r in self.runners]

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
