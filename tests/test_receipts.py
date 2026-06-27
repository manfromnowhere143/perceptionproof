"""Receipts: the audit property must hold — a chain verifies, and any tamper breaks it."""

from __future__ import annotations

from perceptionproof import receipts as R


def _signer() -> R.ReceiptSigner:
    return R.ReceiptSigner.from_seed(b"\x01" * 32)


def _chain(signer: R.ReceiptSigner) -> list[dict]:
    chain: list[dict] = []
    prev = R.GENESIS_HASH
    for i, step in enumerate(["scene.ingest", "perception.run", "score.correlate"]):
        rec = signer.sign_step(
            run_id="run-abc",
            step=step,
            segment_id=f"seg_{i}",
            inputs_hash=f"in{i}",
            outputs_hash=f"out{i}",
            prev_receipt_hash=prev,
        )
        chain.append(rec)
        prev = rec["content_hash"]
    return chain


def test_canonical_json_is_order_independent():
    assert R.canonical_json({"b": 1, "a": 2}) == R.canonical_json({"a": 2, "b": 1})


def test_run_id_is_deterministic_and_changes_with_input():
    a = R.run_id("p", "m", "v", "s")
    assert a == R.run_id("p", "m", "v", "s")
    assert a != R.run_id("p", "m", "v", "s2")
    assert len(a) == 64


def test_valid_chain_verifies():
    s = _signer()
    chain = _chain(s)
    assert R.verify_chain(chain, s.verify_key_hex) is True


def test_tampered_field_breaks_verification():
    s = _signer()
    chain = _chain(s)
    chain[1]["outputs_hash"] = "tampered"
    assert R.verify_chain(chain, s.verify_key_hex) is False


def test_broken_link_breaks_chain():
    s = _signer()
    chain = _chain(s)
    chain[2]["prev_receipt_hash"] = "0" * 64  # wrong predecessor
    assert R.verify_chain(chain, s.verify_key_hex) is False


def test_wrong_key_fails():
    s = _signer()
    chain = _chain(s)
    other = R.ReceiptSigner.from_seed(b"\x02" * 32)
    assert R.verify_chain(chain, other.verify_key_hex) is False
