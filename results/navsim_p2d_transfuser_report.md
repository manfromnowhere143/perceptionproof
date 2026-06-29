# P2d — TransFuser (real camera+LiDAR planner): pipeline validated, result underpowered

Date: 2026-06-28. Compute: GCP n2-standard-16, CPU. Real NAVSIM PDM simulator + a real SOTA
sensor planner.

## What this tested

P2c used a weak ego-status MLP planner whose failures are partly non-collision. The natural
next question: does a label-free signal predict a **real perception planner's** closed-loop
failures? We ran **TransFuser** — a published camera+LiDAR planner — as a **3-seed ensemble**
(pretrained checkpoints, `autonomousvision/navsim_baselines`).

## Two outcomes — one engineering win, one honest power limit

### 1. The real-sensor pipeline is validated (the hard part)

Getting a real sensor planner running end-to-end is non-trivial, and it works: 3 pretrained
TransFuser checkpoints load, camera+LiDAR sensors load, the model runs (CPU), and every
trajectory is scored through the PDM simulator — **396 scenes across 52 drives, 0 errors** (drives
counted by unique OpenScene log id, the cluster-bootstrap unit). The collision-geometry +
ensemble-disagreement signals are computed on real TransFuser trajectories.

(Note: this required the **mini** split. The larger navtrain sensors are frame-version
inconsistent with the OpenScene metadata — conclusively diagnosed, documented in `CONTINUITY.md`
— so they could not be used. The mini split ships metadata+sensors as a matched, frame-consistent
package.)

### 2. The result is underpowered — a strong planner rarely fails

| Gate | Events in 396 scenes |
|---|---|
| Collision (NC < 1) | **3** |
| Off-road (DAC < 1) | **12** |
| Any gate (PDMS = 0) | **53** |

TransFuser is good, so it almost never collides (3 events) or leaves the road (12) on 396 mini
scenes — **far too few to estimate a failure-prediction AUROC.** The collision/off-road CIs are
uninformative (e.g., NC AUROC CI spans ~0.3–0.98). Only the broader any-gate target has borderline
power:

| Signal → any-gate (PDMS=0) | AUROC | 95% CI |
|---|---|---|
| disagreement | 0.600 | [0.511, 0.695] |
| collision_risk | 0.552 | [0.473, 0.626] |
| off_road | 0.516 | [0.495, 0.545] |

Paired matched-vs-baseline differences: all **inconclusive** (CIs include 0). The faint
disagreement→any-gate signal (0.60) is consistent with P2c but cannot be confirmed at this power.
These exact numbers regenerate from the committed `tf_mini_result.json` via
`python experiments/navsim_p2c/analyze.py experiments/navsim_p2c/tf_mini_result.json`.

## Honest conclusion

- **Engineering:** the experiment design runs end-to-end on a real SOTA camera+LiDAR planner — a
  genuine capability, not a toy.
- **Science:** *inconclusive by underpowering*, not by refutation. You cannot measure a
  failure-predictor on a planner that rarely fails over a small dataset. This is the expected,
  honest result — reported, not hidden.
- **What a powered real-sensor result needs:** a much larger *frame-consistent* sensor dataset
  (full trainval sensors, ~TBs, or resolving the navtrain metadata/sensor version mismatch) so
  that even a ~1% failure rate yields ~100+ events. That is a data-engineering task for a future
  session.

## The headline result is unchanged

P2c stands as the powered closed-loop finding: label-free signals predict the binary PDMS gate
events at AUROC ~0.8 across 55 drives (decisively above chance). The TransFuser run extends the
*pipeline* to real sensors and honestly reports that a powered real-planner measurement needs more
consistent data.

## Reproduce

`experiments/navsim_p2c/analyze.py` on the TransFuser output; scoring via `tf_score.py` (3-seed
TransFuser, sensor loader, PDM scoring). No OpenScene/nuPlan data redistributed.
