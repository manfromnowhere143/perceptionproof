"""End-to-end runner. Threads the mission (ingest -> run_models -> signal.compute ->
score.correlate), emits a signed receipt per step, writes a report + receipt chain, and
verifies the chain.

A single backend-driven loop (`_run_backend`) serves both paths:
- run_synthetic: SyntheticBackend (planted data) — proves the machine on known answers.
- run_local: LocalBackend (DatasetAdapter + ModelRunners) — the production path; with real
  adapters/runners it produces the real result, with fixtures it is tested offline.
Only ingest/run_models differ between them; everything downstream is identical.
"""

from __future__ import annotations

import json
import os

import numpy as np

from perceptionproof.backend import LocalBackend
from perceptionproof.backend_synthetic import SyntheticBackend, hash_model_outputs
from perceptionproof.receipts import (
    ReceiptSigner,
    canonical_json,
    run_id as make_run_id,
    sha256_hex,
    verify_chain,
)
from perceptionproof.scoring import failure_mining, risk_coverage, spearman_with_ci
from perceptionproof.signals import s1_ensemble_disagreement

DEV_SEED = b"\x07" * 32  # open repo dev signing key; reproducible, NOT a production secret

SYNTHETIC_NOTE = "SYNTHETIC PLUMBING TEST — correlation is planted by construction; not a scientific finding."
FIXTURE_NOTE = "FIXTURE PIPELINE TEST — fixture adapter/runners; validates the LocalBackend seam, not a finding."


def _run_backend(backend, signer, run_id, segment_ids, sigma, theta_rfs, out_dir, prefix, note):
    receipts: list[dict] = []
    rows: list[dict] = []
    for sid in segment_ids:
        scene = backend.ingest(sid)
        receipts.append(backend.emit_receipt(
            step="scene.ingest", segment_id=sid,
            inputs_hash=sha256_hex(sid.encode()),
            outputs_hash=sha256_hex(canonical_json({"rfs": scene.rfs, "drive": scene.drive_id})),
        ))
        outs = backend.run_models(scene)
        oh = hash_model_outputs(outs)
        receipts.append(backend.emit_receipt(
            step="perception.run", segment_id=sid,
            inputs_hash=sha256_hex(canonical_json({"scene": sid})),
            outputs_hash=oh, extra={"n_models": len(outs)},
        ))
        g1 = s1_ensemble_disagreement(outs, sigma=sigma)
        receipts.append(backend.emit_receipt(
            step="signal.compute", segment_id=sid,
            inputs_hash=oh,
            outputs_hash=sha256_hex(canonical_json({"g1": round(g1, 9)})), extra={"g1": g1},
        ))
        rows.append({"segment_id": sid, "drive_id": scene.drive_id, "rfs": scene.rfs, "g1": g1})

    g = np.array([r["g1"] for r in rows])
    report: dict = {"run_id": run_id, "n": len(rows), "theta_rfs": theta_rfs, "note": note}

    rfs_present = all(r["rfs"] is not None for r in rows)
    if rfs_present:
        rfs = np.array([r["rfs"] for r in rows], dtype=float)
        drives = np.array([r["drive_id"] for r in rows])
        failure = (rfs < theta_rfs).astype(int)
        rc = risk_coverage(g, 10.0 - rfs)
        report["spearman_g1_vs_negRFS"] = spearman_with_ci(g, -rfs, drives, n_boot=500, n_perm=500)
        report["failure_mining_g1"] = failure_mining(g, failure)
        report["risk_coverage_g1"] = {"aurc": rc["aurc"], "e_aurc": rc["e_aurc"]}
    else:
        # e.g. NAVSIM scenes carry PDMS, not RFS — wire PDMS scoring at P2.
        report["rfs_available"] = False
        report["g1_stats"] = {"mean": float(g.mean()), "std": float(g.std()), "n": int(g.size)}

    summary_inputs = [{"segment_id": r["segment_id"], "drive_id": r["drive_id"],
                       "rfs": r["rfs"], "g1": round(r["g1"], 9)} for r in rows]
    receipts.append(backend.emit_receipt(
        step="score.correlate", segment_id=None,
        inputs_hash=sha256_hex(canonical_json(summary_inputs)),
        outputs_hash=sha256_hex(canonical_json(report)),
    ))

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{prefix}_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(out_dir, f"{prefix}_receipts.jsonl"), "w") as f:
        for r in receipts:
            f.write(json.dumps(r) + "\n")

    if not verify_chain(receipts, signer.verify_key_hex):
        raise RuntimeError("receipt chain failed verification")
    return report, receipts, signer.verify_key_hex


def default_synthetic_slice(n_drives: int = 30, per_drive: int = 2) -> list[str]:
    return [f"drive{d:02d}_seg{j}" for d in range(n_drives) for j in range(per_drive)]


def run_synthetic(segment_ids: list[str], sigma: float = 1.0, theta_rfs: float = 5.0,
                  out_dir: str = "results", seed: int = 20260627):
    rid = make_run_id("synthetic-protocol", "synthetic-models", "0.1.0", sha256_hex(canonical_json(segment_ids)))
    signer = ReceiptSigner.from_seed(DEV_SEED)
    backend = SyntheticBackend(rid, signer, seed=seed)
    return _run_backend(backend, signer, rid, segment_ids, sigma, theta_rfs, out_dir, "synthetic", SYNTHETIC_NOTE)


def run_local(adapter, runners, sigma: float = 1.0, theta_rfs: float = 5.0,
              out_dir: str = "results", note: str = FIXTURE_NOTE):
    """Run the study over any DatasetAdapter + ModelRunners (the production path).

    With real adapters/runners this is the real study; with fixtures it tests the seam.
    """
    segment_ids = adapter.segment_ids()
    model_ids = [getattr(r, "model_id", "?") for r in runners]
    rid = make_run_id(
        "local-protocol",
        sha256_hex(canonical_json(model_ids)),
        "0.1.0",
        sha256_hex(canonical_json(segment_ids)),
    )
    signer = ReceiptSigner.from_seed(DEV_SEED)
    backend = LocalBackend(rid, adapter, runners, signer)
    return _run_backend(backend, signer, rid, segment_ids, sigma, theta_rfs, out_dir, "local", note)


def verify_receipts_file(path: str) -> bool:
    with open(path) as f:
        receipts = [json.loads(line) for line in f if line.strip()]
    verify_key_hex = ReceiptSigner.from_seed(DEV_SEED).verify_key_hex
    return verify_chain(receipts, verify_key_hex)
