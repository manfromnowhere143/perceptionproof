# P2a — First real measurement on NAVSIM

Date: 2026-06-28. Compute: GCP CPU VM (8 vCPU), no GPU, no sensor data, no metric cache.

## Question

Does label-free ensemble **disagreement** predict where driving models make **errors**, on real
autonomous-driving frames?

## Setup (real data, real models)

- Data: OpenScene / nuPlan **trainval** logs (the real NAVSIM scene data), ego-status + human
  future trajectories. No camera/LiDAR sensors used; ego-status agents need none.
- Model: a 4-member **ego-status MLP** ensemble (8-dim ego state → 10-pose future), each member a
  different random seed, trained for 400 epochs.
- Split: **disjoint by log** — 99 logs, 3,212 train / 788 test scenes, 20 test logs. No leakage.
- Signal: S1 ensemble trajectory disagreement (MMD, median-heuristic σ). Outcome: open-loop ADE of
  the members vs the human future. All statistics computed by PerceptionProof's unit-tested
  `signals.py` / `scoring.py`.

## Result

| Metric | Value |
|---|---|
| Spearman ρ (disagreement vs ADE), cluster-bootstrap by log | **0.699**, 95% CI [0.599, 0.750], p = 0.0005, n = 788 |
| Failure mining AUROC / AP (failure = ADE > median) | **0.855** / 0.846 |
| precision@50 / @100 | 0.98 / 0.95 |
| Selective prediction AURC / E-AURC | 0.563 / 0.180 |
| ADE mean / median; FDE mean | 1.03 m / 0.62 m; 2.74 m |

Label-free disagreement strongly tracks where these models err on real, held-out scenes.

## Strengthening: independent-outcome (leave-one-out) test

To check that the result is not just the algebraic variance↔mean-error coupling of an ensemble,
we re-ran with the signal and outcome drawn from **disjoint** models: disagreement among members
{0,1,2} as the signal, and the open-loop error of the **held-out** member 3 as the outcome.

| Metric | Value |
|---|---|
| Spearman ρ (disagreement{0,1,2} vs error of held-out member 3) | **0.683**, 95% CI [0.589, 0.729], p = 0.0005 |
| Failure mining AUROC / AP | **0.844** / 0.832 |

Essentially unchanged from the coupled version (0.699 / 0.855). So disagreement tracks genuine
**scene difficulty**, not an algebraic artifact. Reproduce: `experiments/navsim_p2a/leave_one_out.py`.
This retires caveat 3 below. The result is still open-loop (caveat 1 stands).

## Honest caveats (what this is NOT, yet)

1. **Open-loop, not closed-loop.** The outcome is ADE vs the human trajectory, not PDMS or a
   safety-aligned score. The thesis's decisive claim is about closed-loop/human-rated outcomes —
   that is P2b (NAVSIM PDMS) and the WOD-E2E RFS result, not this.
2. **Weak models.** Ego-status-only MLPs (no scene perception). Disagreement predicting the error
   of weak models is an easier setting than SOTA planners.
3. **Structural coupling.** ~~For an ensemble, disagreement and mean error are partly linked by
   construction.~~ ADDRESSED by the leave-one-out test above (disjoint signal/outcome models →
   ρ = 0.683 holds). The signal reflects scene difficulty, not an algebraic artifact.
4. **Not the official navtest split.** This is our own by-log split over OpenScene trainval; the
   official navtest benchmark needs the separate OpenScene test download (planned).

## Provenance

Trajectories produced on real frames on the VM; correlation computed locally by the tested
scoring code. Raw per-scene trajectories are not committed (OpenScene-derived data is not
redistributed; see `DATA_LICENSES.md`) — only these aggregate numbers.

## Next (P2b)

Independent outcome: compute NAVSIM **PDMS** per scene (metric cache + PDM simulator) and correlate
disagreement against it; then the WOD-E2E **RFS** result. That is the closed-loop-aligned,
non-structural test the thesis actually rests on.
