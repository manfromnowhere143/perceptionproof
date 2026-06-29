# P2e — WOD-E2E Rater Feedback Score: the label-free signal weakly predicts human ratings (H1 not met)

Date: 2026-06-29. Compute: GCP n2-standard-16, CPU. Data: Waymo Open Dataset
End-to-End (WOD-E2E) validation split — the official long-tail human-rated benchmark.
Scored by Waymo's own `rater_feedback_utils.get_rater_feedback_score` and by the
unit-tested statistics in this repo.

## What this tested

The previous phases used surrogate targets (open-loop ADE, the PDM simulator score,
the binary PDMS gates). P2e tests the real thing the field actually cares about:
**human raters' judgment of long-tail driving.** Waymo's Rater Feedback Score (RFS)
grades a proposed 5-second trajectory against rater-preferred trajectories, each with
a human quality label, inside a speed-scaled trust region (arXiv 2510.26125).

The pre-registered question (H1): **does a cheap, label-free signal predict per-frame RFS?**
Confirmed only if Spearman ρ ≥ 0.30 with BH-corrected q < 0.05.

## Method

- Parsed all **93 validation shards** (243 GB; each `E2EDFrame` embeds eight camera
  images, so the parse was fanned across the 16 cores).
- Trained a **4-seed ego-status MLP ensemble** (sklearn, 18,600 frames) to predict the
  20-waypoint future trajectory from the current ego state + driving intent — **no
  sensors, no labels**. Rater-labeled frames were held out of training.
- On each of the **479 rater-labeled frames**: predicted four trajectories, measured
  **ensemble disagreement** (mean pairwise displacement — the label-free signal), scored
  the ensemble-mean trajectory with the official RFS, and recorded **ADE vs the human
  future** as an oracle anchor (ADE requires the human label, so it is *not* the cheap
  signal — it only certifies that RFS is genuinely predictable from trajectory quality).
- All correlations target `neg_RFS = −RFS` (positive ρ ⇒ signal flags *worse* ratings),
  with **drive-level (per-shard) cluster-bootstrap** 95% CIs (10k resamples).

## Result

479 rater frames across 93 drives. Minimum detectable |ρ| at this n (power 0.8) = 0.128.
RFS distribution: mean 7.41, sd 2.24, range [3.0, 10.0] — well spread, not degenerate.

| Signal | needs label? | Spearman ρ vs neg-RFS | 95% CI (cluster) | H1 (ρ≥0.3) |
|---|---|---|---|---|
| **ensemble disagreement** | **no (label-free)** | **+0.151** | [+0.063, +0.237] | **not met** |
| ADE-to-human (oracle anchor) | yes | +0.395 | [+0.326, +0.458] | met |

Triage of the worst-rated frames (failure = bottom-quartile RFS, base rate 0.25):

| Signal | AUROC | AP (base 0.25) | E-AURC |
|---|---|---|---|
| disagreement | 0.629 | 0.390 | 1.478 |
| ADE (oracle) | 0.734 | 0.421 | 0.989 |

Open-loop coupling here: disagreement vs ADE ρ = +0.185 [+0.095, +0.273].

## Honest reading

1. **RFS is a real, predictable target.** Trajectory error (ADE) tracks the human score
   at ρ = 0.40 — humans rate trajectory quality coherently, and a label-requiring oracle
   recovers it. This validates the benchmark and the pipeline end-to-end against Waymo's
   own scorer.
2. **The cheap label-free signal predicts human ratings — but weakly.** Disagreement
   correlates with worse RFS at ρ = 0.151; the CI excludes zero and it survives BH-FDR
   (q < 0.05), so the effect is **real, not noise**. But it sits **below the pre-registered
   ρ ≥ 0.30 bar**, so **H1 is not confirmed** on WOD-E2E. This is a published negative.
3. **Why weak — and the honest lever.** The planner is **ego-status only** (12 features:
   ego kinematics + intent, no perception). Its members disagree mostly on ego dynamics,
   not on scene difficulty — here disagreement↔ADE is only ρ = 0.19, versus ρ = 0.70 with
   the richer P2a setup. A perception-grounded ensemble (cameras/LiDAR) should carry more
   scene information into the disagreement signal. The weak result is consistent with a
   **weak planner**, not a refutation of the hypothesis for stronger planners.

## Where this leaves the arc

| Target | Signal behavior |
|---|---|
| Open-loop ADE (P2a) | strong — ρ = 0.70 |
| Closed-loop PDMS **score** (P2b) | null — ρ ≈ 0 |
| Closed-loop **gate events** (P2c) | decisive — AUROC ~0.8 |
| Real-sensor TransFuser gates (P2d) | underpowered (strong planner rarely fails) |
| **Human RFS (P2e)** | **real but weak — ρ = 0.15, below the 0.3 bar** |

The label-free signal is strongest on the cheapest target (open-loop error) and
progressively weaker as the target moves toward human and closed-loop judgment — except
on the binary safety **gates**, where it is decisive. Against the actual human raters, an
ego-only ensemble is honestly insufficient. The next power lever is a perception-grounded
ensemble, not a different statistic.

## Reproduce

```bash
# on a CPU VM with the WOD-E2E val split + the waymo-open-dataset toolkit (see DATA_LICENSES.md):
python experiments/wod_e2e_rfs/validate_rfs_api.py   # confirm the official RFS call on one frame
python experiments/wod_e2e_rfs/run_rfs.py            # parse 93 shards, train ensemble, score RFS -> wod_rfs_out.json
# locally, with the repo's tested statistics:
python experiments/wod_e2e_rfs/analyze.py experiments/wod_e2e_rfs/wod_rfs_out.json
```

No Waymo frames are redistributed — only the per-frame derived scalars
(`disagreement`, `rfs`, `ade`, `init_speed`, `shard`) in `wod_rfs_out.json`.
