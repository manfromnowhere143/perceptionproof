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

## Honest caveats (what this is NOT, yet)

1. **Open-loop, not closed-loop.** The outcome is ADE vs the human trajectory, not PDMS or a
   safety-aligned score. The thesis's decisive claim is about closed-loop/human-rated outcomes —
   that is P2b (NAVSIM PDMS) and the WOD-E2E RFS result, not this.
2. **Weak models.** Ego-status-only MLPs (no scene perception). Disagreement predicting the error
   of weak models is an easier setting than SOTA planners.
3. **Structural coupling.** For an ensemble, disagreement (variance) and mean error are partly
   linked by construction; this measurement confirms the pipeline and that the signal is real on
   real data, but is not yet an independent-outcome test.
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
