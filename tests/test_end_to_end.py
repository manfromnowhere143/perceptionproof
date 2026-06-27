"""The whole machine runs end-to-end on synthetic (planted) data, and the receipt
chain verifies. Proves plumbing, not a scientific finding."""

from __future__ import annotations

from harness.runner import run_synthetic, verify_receipts_file
from perceptionproof.receipts import verify_chain


def test_synthetic_pipeline_runs_and_chain_verifies(tmp_path):
    ids = [f"drive{d:02d}_seg{j}" for d in range(15) for j in range(2)]  # 30 segments, 15 drives
    report, receipts, vk = run_synthetic(ids, out_dir=str(tmp_path))

    assert verify_chain(receipts, vk)
    assert report["n"] == 30
    # 4 receipts per... no: 3 per segment + 1 final score receipt
    assert len(receipts) == 30 * 3 + 1
    # planted relationship: more disagreement <-> lower RFS -> positive spearman, separable failures
    assert report["spearman_g1_vs_negRFS"]["rho"] > 0.3
    assert report["failure_mining_g1"]["auroc"] > 0.6
    assert report["note"].startswith("SYNTHETIC")


def test_written_receipts_file_verifies(tmp_path):
    ids = [f"drive{d:02d}_seg0" for d in range(12)]
    run_synthetic(ids, out_dir=str(tmp_path))
    assert verify_receipts_file(str(tmp_path / "synthetic_receipts.jsonl")) is True
