"""Tamper-evident provenance: canonical hashing, Ed25519 signing, and chain
verification. This is the audit property an outside engineer uses to trust a run
(docs/ARCHITECTURE.md sec 5). Pure CPU, no dataset or GPU required.

A receipt commits to (run_id, step, inputs_hash, outputs_hash, prev_receipt_hash).
content_hash = sha256(canonical_json(content)); the chain links via prev_receipt_hash;
the signature is Ed25519 over content_hash. Verification recomputes the hash, checks
the signature, and walks the chain — all without any private material.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from nacl.encoding import HexEncoder
from nacl.signing import SigningKey, VerifyKey

GENESIS_HASH = "0" * 64

_CONTENT_FIELDS = (
    "run_id",
    "step",
    "segment_id",
    "inputs_hash",
    "outputs_hash",
    "prev_receipt_hash",
    "extra",
)


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON encoding: sorted keys, no insignificant whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_hash(content: dict) -> str:
    return sha256_hex(canonical_json(content))


def run_id(protocol_hash: str, models_lock_hash: str, code_version: str, slice_hash: str) -> str:
    """Content-addressed run id (blake2b of the four pinned inputs)."""
    h = hashlib.blake2b(digest_size=32)
    for part in (protocol_hash, models_lock_hash, code_version, slice_hash):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class ReceiptSigner:
    """Holds an Ed25519 signing key and emits signed, chained receipt dicts."""

    def __init__(self, signing_key: SigningKey) -> None:
        self._sk = signing_key

    @classmethod
    def from_seed(cls, seed: bytes) -> "ReceiptSigner":
        """Deterministic signer from a 32-byte seed (used for reproducible runs/tests)."""
        return cls(SigningKey(seed))

    @property
    def verify_key_hex(self) -> str:
        return self._sk.verify_key.encode(HexEncoder).decode()

    def sign_step(
        self,
        *,
        run_id: str,
        step: str,
        inputs_hash: str,
        outputs_hash: str,
        prev_receipt_hash: str,
        segment_id: str | None = None,
        extra: dict | None = None,
    ) -> dict:
        content = {
            "run_id": run_id,
            "step": step,
            "segment_id": segment_id,
            "inputs_hash": inputs_hash,
            "outputs_hash": outputs_hash,
            "prev_receipt_hash": prev_receipt_hash,
            "extra": extra or {},
        }
        ch = content_hash(content)
        signature = self._sk.sign(ch.encode("utf-8")).signature.hex()
        return {**content, "content_hash": ch, "signature": signature}


def verify_receipt(receipt: dict, verify_key_hex: str) -> bool:
    """Recompute the content hash and verify the Ed25519 signature."""
    content = {k: receipt.get(k) for k in _CONTENT_FIELDS}
    if content_hash(content) != receipt.get("content_hash"):
        return False
    try:
        VerifyKey(verify_key_hex, HexEncoder).verify(
            receipt["content_hash"].encode("utf-8"), bytes.fromhex(receipt["signature"])
        )
    except Exception:
        return False
    return True


def verify_chain(receipts: list[dict], verify_key_hex: str) -> bool:
    """Verify every receipt and that prev_receipt_hash links form an unbroken chain."""
    prev = GENESIS_HASH
    for r in receipts:
        if r.get("prev_receipt_hash") != prev:
            return False
        if not verify_receipt(r, verify_key_hex):
            return False
        prev = r["content_hash"]
    return True
