# P2 Setup — wiring real models on real frames

P2 produces the first real numbers. The harness, signals, statistics, and receipts are
already built and tested (`pytest`), and the production backend (`LocalBackend`) is wired:
P2 only implements two real adapters and the real model runners, then runs the proven
pipeline. This document is the exact, researched path.

Datasets are license-gated and are not redistributed by this repo (see `DATA_LICENSES.md`).

## Two tracks

### Track A — NAVSIM first (fastest real signal: disagreement vs PDMS)

NAVSIM ships multiple baseline agents over real OpenScene/nuPlan frames, so one framework
yields an ensemble (disagreement) and a closed-loop-aligned score (PDMS) immediately —
the quickest honest test of "does disagreement predict low PDMS on the long tail."

1. `git clone https://github.com/autonomousvision/navsim`; create its conda env per the repo README.
2. Download OpenScene (navtest split) + maps; set the `NUPLAN_*` / `OPENSCENE_*` env vars the devkit expects. (Non-commercial research license.)
3. Run >= 2 baseline agents — `transfuser_agent`, `ego_status_mlp_agent`, `constant_velocity_agent` — to produce a trajectory per token, and the PDM Score per token.
4. Implement `NavsimAdapter.segment_ids/load` (enumerate navtest tokens; return `SceneBundle` with the token's ego/route; attach PDMS via the report side-channel) and wrap each agent as a `NavsimAgentRunner` (`perceptionproof/models.py`).
5. `run_local(NavsimAdapter(...), [NavsimAgentRunner(...), ...])` → S1 disagreement per scene. Wire a PDMS-scoring path and test: does disagreement rank-correlate with low PDMS (and recover ranking vs the open-loop baseline, H2)?

GPU: one L4/A100 is sufficient for inference over a navtest slice.

### Track B — WOD-E2E (the H1 result: signal vs human RFS)

This is the headline result — does a label-free signal predict the human Rater Feedback Score on long-tail segments.

1. Accept the Waymo Open Dataset license and download WOD-E2E **validation** (8-camera segments + ego + routing + per-trajectory RFS) at https://waymo.com/open.
2. Implement `WodE2EAdapter.segment_ids/load` to decode segments + RFS.
3. Provide the model runners as an ensemble: a camera E2E planner (`E2EPlannerRunner`), and — to exercise S3/S4 — an occupancy model and a VLA (`AutoVLA`). Pin each into `protocol/models.lock.json` with weight SHA-256.
4. **Freeze the `TBD@P2` values in `PREREGISTRATION.md` from the real RFS distribution BEFORE scoring**: `theta_RFS`, `theta_occ`, `lambda`, slice size `n`, and the resulting minimum-detectable `rho`.
5. `run_local(WodE2EAdapter(...), runners)` → S1–S4 vs RFS → H1/H3, plus the selective-prediction AURC (the deployable triage number).

## Compute

A single modern GPU (L4 or A100) handles inference over the chosen slice for both tracks.
Provision a GPU VM on the team's existing cloud estate (internal ops doc holds the
project-specific provisioning; this repo stays cloud-agnostic). CUDA 12 + a PyTorch build
matching each model repo.

## Order of operations (discipline)

1. Pin models + freeze the slice ids and pre-registration values **before** computing any correlation.
2. Run the proven pipeline; every step emits a signed receipt.
3. Report all signals, including those that fail; a null result is published unmodified.
4. `python -m harness.cli verify <receipts.jsonl>` must return `VERIFIED` for the run to count.

## Definition of done for P2

A clean run produces receipted `ModelOutput` for the frozen slice on real frames, the
real signal-vs-RFS (and disagreement-vs-PDMS) numbers are computed with confidence
intervals, the receipt chain verifies, and re-running yields the same `run_id` and
byte-identical outputs.
