# Project Status & Roadmap

The durable state of PerceptionProof: what is built, what is verified, and what is next.
Append dated entries; do not rewrite history.

## What this project is

A reproducible study and harness testing whether cheap, label-free signals (model
disagreement, temporal inconsistency, occupancy conflict, VLA reasoning self-consistency)
predict human-rated long-tail driving failure (Waymo's Rater Feedback Score), and whether
they recover the safety ranking on cases where open-loop planning metrics are known to
mis-rank closed-loop driving. It is an evaluation layer **around** the driving stack — not
a controller. The science is open and checkable; orchestration is pluggable via a backend
interface (an open deterministic backend ships here; a separate governed backend is the
commercial half — open-core).

## Contribution, stated honestly

- **Not novel, and not claimed:** disagreement-as-uncertainty (Deep Ensembles, 2017).
- **The contribution:** (1) bridging cheap label-free signals to the 2026 metric-validity
  result that open-loop metrics mis-rank closed-loop safety; (2) a falsifiable, multi-signal
  study under one pre-registered protocol; (3) auditable, signed provenance for the whole
  evaluation. The study is publishable even if its result is negative.

## Source-of-truth documents

- Mathematics (every signal + validity metric): `docs/MATHEMATICS.md`
- Architecture (backend interface, receipts, mission DAG): `docs/ARCHITECTURE.md`
- Frozen analysis (hypotheses, thresholds, seed): `PREREGISTRATION.md`
- Data licensing & compliance: `DATA_LICENSES.md`

## Phase ledger

| Phase | Gate | State |
|---|---|---|
| P0 Plan | design approved | DONE |
| P1 Scaffold | hypotheses frozen pre-result | DONE — repo skeleton, math, architecture, pre-registration |
| P2 Data & models | inputs reproducible from receipts | NEXT — the only GPU/dataset-bound work |
| P3 Signals | tests green; deterministic | DONE — all four signals implemented + tested on known-answer inputs (S1 MMD disagreement, S2 SE(2)-aligned temporal flicker, S3 corridor occupancy-conflict, S4 semantic-entropy clustering) |
| P4 Study | each finding adversarially verified | core DONE — all validity statistics implemented + tested; awaits real RFS labels |
| P5 Report & repo | reproduce-in-one-command verified | pending |

## What runs today (CPU, no GPU/dataset)

```bash
python -m venv .venv && ./.venv/bin/pip install -e .
./.venv/bin/pip install pytest
./.venv/bin/pytest                                            # 31 tests green
./.venv/bin/python -m harness.cli run --backend synthetic     # full mission, writes results/ + receipts
./.venv/bin/python -m harness.cli verify results/synthetic_receipts.jsonl   # -> VERIFIED
```

Implemented + tested: the receipt layer (hash-chain + Ed25519 sign/verify), all four
signals, all validity statistics (Spearman + cluster bootstrap + permutation null,
failure-mining AUROC/AP, risk-coverage/AURC, Kendall-inversion recovery, mutual
information, BH-FDR), and a synthetic end-to-end run.

The synthetic report is a **plumbing test** (its correlation is planted by construction)
— it confirms the machine threads signals → labels → receipts correctly. It is not a
scientific finding.

## Next (P2)

1. Register and obtain WOD-E2E (RFS labels) and NAVSIM navtest.
2. Pin >= 3 public models (an E2E planner, an occupancy/BEV model, a VLA) into
   `protocol/models.lock.json` with weight hashes.
3. Freeze the long-tail slice ids and the `TBD@P2` pre-registration values **before**
   computing any correlation.
4. Implement `LocalBackend.ingest`/`run_models` to emit real `ModelOutput` + receipts.

Gate to leave P2: a clean run produces receipted `ModelOutput` for the slice, and
re-running yields the same `run_id` and byte-identical outputs.

## Published

Public repository: https://github.com/manfromnowhere143/perceptionproof (Apache-2.0).
CI (ruff + pytest) runs on every push and pull request.

## Log

- **2026-06-27** — P0/P1 complete (thesis, math, architecture, pre-registration). CPU core
  built and green: receipts, all four signals, all validity statistics, and a synthetic
  end-to-end harness + CLI that runs the mission and verifies the receipt chain (31 tests).
  Remaining work is P2 (real models on real frames).
