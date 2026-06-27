"""The production-shaped LocalBackend (DatasetAdapter + ModelRunners + receipts) threads
real-shaped data end-to-end on fixtures, and the receipt chain verifies. Proves the seam
that P2 fills with real adapters/runners."""

from __future__ import annotations

from harness.runner import run_local
from perceptionproof.dataset import FixtureDatasetAdapter
from perceptionproof.models import FixtureModelRunner
from perceptionproof.receipts import verify_chain


def test_local_backend_threads_pipeline_and_verifies(tmp_path):
    ids = [f"drive{d:02d}_seg{j}" for d in range(15) for j in range(2)]  # 30 segments, 15 drives
    adapter = FixtureDatasetAdapter(ids)
    runners = [FixtureModelRunner(i) for i in range(3)]
    report, receipts, vk = run_local(adapter, runners, out_dir=str(tmp_path))

    assert verify_chain(receipts, vk)
    assert report["n"] == 30
    assert len(receipts) == 30 * 3 + 1
    # planted: disagreement tracks low RFS through the real composition path
    assert report["spearman_g1_vs_negRFS"]["rho"] > 0.3
    assert report["failure_mining_g1"]["auroc"] > 0.6
    assert report["note"].startswith("FIXTURE")


def test_local_backend_handles_missing_rfs(tmp_path):
    # An adapter that returns scenes without RFS (NAVSIM-like) must not crash; it reports
    # signal stats and notes RFS is unavailable.
    class NoRfsAdapter(FixtureDatasetAdapter):
        def load(self, segment_id):
            scene = super().load(segment_id)
            scene.rfs = None
            return scene

    adapter = NoRfsAdapter([f"d{j}_seg0" for j in range(8)])
    runners = [FixtureModelRunner(i) for i in range(3)]
    report, receipts, vk = run_local(adapter, runners, out_dir=str(tmp_path))
    assert verify_chain(receipts, vk)
    assert report["rfs_available"] is False
    assert "g1_stats" in report
