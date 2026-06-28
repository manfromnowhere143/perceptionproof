# P2b — Closed-loop PDMS on NAVSIM (a null result, reported unmodified)

Date: 2026-06-28. Compute: GCP CPU VM (8 vCPU), no GPU. NAVSIM PDM simulator.

## Question

The open-loop result (P2a) showed disagreement predicts trajectory error (ρ = 0.699), and the
leave-one-out test showed this is genuine scene difficulty, not an artifact (ρ = 0.683). **Does
that signal transfer to a closed-loop, simulator-derived outcome?** This is the decisive,
non-structural test.

## Setup (real closed-loop simulation)

- 400 real NAVSIM scenes with precomputed **metric caches** (PDM simulator state).
- A native-architecture ego-status MLP ensemble (K = 4) trained on disjoint scenes.
- Outcome: **PDMS** of the deployed planner (member 0) from NAVSIM's PDM simulator — collisions,
  drivable-area compliance, time-to-collision, progress, comfort. Signal: S1 ensemble disagreement.
- Statistics: the unit-tested `perceptionproof.scoring`, cluster-bootstrapped by driving log.

## Result

| | Open-loop (ADE, P2a) | **Closed-loop (PDMS, this)** |
|---|---|---|
| Spearman ρ (disagreement vs outcome) | 0.699 [0.599, 0.750] | **−0.074 [−0.396, 0.285]** |
| Cluster-bootstrap CI | excludes 0 | **includes 0** |
| Failure-mining AUROC | 0.855 | **0.527 (≈ chance)** |
| n / drives | 788 / 20 | 400 / 8 |

**The open-loop signal does not transfer to closed-loop PDMS.** Across 8 drives the correlation is
statistically indistinguishable from zero. (Partial slices swung from ρ = −0.28 at 4 drives to
+0.25 at 6 drives — drive-specific noise that averaged out, which is exactly why the per-drive
cluster bootstrap, not a pooled p-value, is the honest test.)

## Why this matters (it is on-thesis, not a failure)

This is a first-hand, in-pipeline instance of the precise problem that motivates PerceptionProof:
**a signal that looks excellent on an open-loop metric fails to predict the closed-loop /
safety-aligned outcome** — the open-loop↔closed-loop gap (arXiv 2605.00066). We did not have to
look for it across papers; our own cheap signal reproduced it. The pre-registration commits us to
publishing this null unmodified, and we do.

## Honest caveats (bounding the claim)

1. **Weak deployed planner.** Ego-status-only MLP; its PDMS is often dominated by collision /
   off-road events (PDMS is near-bimodal: many 0.0 and 1.0) that trajectory *spread* may simply not
   encode.
2. **Power.** 8 drives → a wide cluster CI. This is "no evidence of transfer," not "proven zero
   transfer"; a small effect cannot be excluded.
3. **One signal.** This tests S1 (trajectory disagreement) only. S3 (occupancy conflict) is designed
   around collision/occlusion risk — the very thing PDMS penalizes — and is the natural next signal
   to test for closed-loop transfer. The S1 null motivates exactly that.

## Reproduce

`experiments/navsim_p2b/` — metric caching, `score_pdms.py`, `analyze_pdms.py`. No OpenScene/nuPlan
data redistributed; aggregate numbers only.

## Takeaway

Open-loop predictivity (ρ = 0.70) is real and robust; closed-loop transfer (ρ ≈ −0.07) is not
demonstrated for S1. The right next step is signal diversity (S3/S4) and a stronger deployed
planner — not a louder claim about S1.
