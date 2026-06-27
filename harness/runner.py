"""End-to-end runner. Threads the mission (ingest -> run_models -> signal.compute ->
score.correlate), emits a signed receipt per step, writes a report + receipt chain,
and asserts the chain verifies.

run_synthetic uses SyntheticBackend (planted data) to prove the machine works on
known-answer inputs. The real run swaps in LocalBackend/MaestroBackend at P2+ with
identical wiring; only ingest/run_models change.
"""

from __future__ import annotations

import json
import os

import numpy as np

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


def default_synthetic_slice(n_drives: int = 30, per_drive: int = 2) -> list[str]:
    return [f"drive{d:02d}_seg{j}" for d in range(n_drives) for j in range(per_drive)]


def run_synthetic(segment_ids: list[str], sigma: float = 1.0, theta_rfs: float = 5.0,
                  out_dir: str = "results", seed: int = 20260627):
    rid = make_run_id("synthetic-protocol", "synthetic-models", "0.1.0", sha256_hex(canonical_json(segment_ids)))
    signer = ReceiptSigner.from_seed(DEV_SEED)
    backend = SyntheticBackend(rid, signer, seed=seed)

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
    rfs = np.array([r["rfs"] for r in rows])
    drives = np.array([r["drive_id"] for r in rows])
    failure = (rfs < theta_rfs).astype(int)
    rc = risk_coverage(g, 10.0 - rfs)

    report = {
        "run_id": rid,
        "n": len(rows),
        "theta_rfs": theta_rfs,
        "spearman_g1_vs_negRFS": spearman_with_ci(g, -rfs, drives, n_boot=500, n_perm=500),
        "failure_mining_g1": failure_mining(g, failure),
        "risk_coverage_g1": {"aurc": rc["aurc"], "e_aurc": rc["e_aurc"]},
        "note": SYNTHETIC_NOTE,
    }

    summary_inputs = [{"segment_id": r["segment_id"], "drive_id": r["drive_id"],
                       "rfs": r["rfs"], "g1": round(r["g1"], 9)} for r in rows]
    receipts.append(backend.emit_receipt(
        step="score.correlate", segment_id=None,
        inputs_hash=sha256_hex(canonical_json(summary_inputs)),
        outputs_hash=sha256_hex(canonical_json(report)),
    ))

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "synthetic_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(out_dir, "synthetic_receipts.jsonl"), "w") as f:
        for r in receipts:
            f.write(json.dumps(r) + "\n")

    if not verify_chain(receipts, signer.verify_key_hex):
        raise RuntimeError("receipt chain failed verification")
    return report, receipts, signer.verify_key_hex


def verify_receipts_file(path: str) -> bool:
    with open(path) as f:
        receipts = [json.loads(line) for line in f if line.strip()]
    verify_key_hex = ReceiptSigner.from_seed(DEV_SEED).verify_key_hex
    return verify_chain(receipts, verify_key_hex)
