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
| P2 Data & models | inputs reproducible from receipts | IN PROGRESS — production `LocalBackend` + `DatasetAdapter`/`ModelRunner` seam BUILT + tested on fixtures; real WOD-E2E/NAVSIM adapters + model runners remain (GPU/dataset-bound). Runbook: `docs/P2_SETUP.md` |
| P3 Signals | tests green; deterministic | DONE — all four signals implemented + tested on known-answer inputs (S1 MMD disagreement, S2 SE(2)-aligned temporal flicker, S3 corridor occupancy-conflict, S4 semantic-entropy clustering) |
| P4 Study | each finding adversarially verified | core DONE — all validity statistics implemented + tested; awaits real RFS labels |
| P5 Report & repo | reproduce-in-one-command verified | pending |

## What runs today (CPU, no GPU/dataset)

```bash
python -m venv .venv && ./.venv/bin/pip install -e ".[dev]"
./.venv/bin/pytest                                            # 33 tests green
./.venv/bin/python -m harness.cli run --backend synthetic     # full mission, writes results/ + receipts
./.venv/bin/python -m harness.cli verify results/synthetic_receipts.jsonl   # -> VERIFIED
```

Implemented + tested: the receipt layer (hash-chain + Ed25519 sign/verify), all four
signals, all validity statistics (Spearman + cluster bootstrap + permutation null,
failure-mining AUROC/AP, risk-coverage/AURC, Kendall-inversion recovery, mutual
information, BH-FDR), a synthetic end-to-end run, AND the production `LocalBackend`
(DatasetAdapter + ModelRunner composition) validated end-to-end on fixtures.

The synthetic report is a **plumbing test** (its correlation is planted by construction)
— it confirms the machine threads signals → labels → receipts correctly. It is not a
scientific finding.

## Next (P2) — see docs/P2_SETUP.md for the full researched runbook

The harness, signals, statistics, receipts, and the production `LocalBackend` are built and
tested. P2 only fills the real adapters/runners (genuinely GPU/dataset-bound):

1. Track A (fastest): NAVSIM — implement `NavsimAdapter` + `NavsimAgentRunner`; run >= 2
   baseline agents over navtest; test disagreement vs PDMS.
2. Track B (the H1 result): WOD-E2E val — implement `WodE2EAdapter` (cameras+ego+routing+RFS);
   ensemble of E2E planner + occupancy + VLA runners; signals S1–S4 vs RFS.
3. Freeze the `TBD@P2` pre-registration values from the real RFS/PDMS distribution BEFORE scoring.

Gate to leave P2: a clean run produces receipted `ModelOutput` for the slice on real frames,
the real numbers are computed with CIs, the chain verifies, and re-running is byte-identical.

## Published

Public repository: https://github.com/manfromnowhere143/perceptionproof (Apache-2.0).
CI (ruff + pytest) runs on every push and pull request.

## First real measurement (P2a)

On 788 held-out real NAVSIM scenes (disjoint by-log split, 4-member ego-status MLP ensemble,
CPU on GCP), label-free disagreement predicts open-loop error: Spearman ρ=0.699 [0.599, 0.750],
p=0.0005; failure-mining AUROC=0.855 / AP=0.846 (prec@50=0.98); E-AURC=0.180. Full writeup +
honest caveats (open-loop not closed-loop; weak ego-status models; structural coupling; own split
not official navtest): `results/navsim_p2a_report.md`. Next decisive test = P2b (NAVSIM PDMS) and
WOD-E2E RFS. VM bootstrap scripts: scratchpad (vm_setup.sh, vm_data.sh, pp_experiment.py).

## Log

- **2026-06-28** — P2a: first REAL number on real NAVSIM data (ρ=0.70, AUROC=0.855). NAVSIM env +
  minimal data (maps + trainval logs, no sensors) stood up on GCP CPU VM engineering-node-02;
  4-member ego-status MLP ensemble trained; disagreement-vs-ADE scored by the tested code.
- **2026-06-27** — P0/P1 complete (thesis, math, architecture, pre-registration). CPU core
  built and green: receipts, all four signals, all validity statistics, and a synthetic
  end-to-end harness + CLI that runs the mission and verifies the receipt chain (31 tests).
  Remaining work is P2 (real models on real frames).
