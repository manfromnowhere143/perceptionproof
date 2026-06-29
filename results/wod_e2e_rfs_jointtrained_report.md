# P2h — jointly-trained vision ensemble (GPU): disagreement still does not predict human RFS

Date: 2026-06-29. Compute: GCP L4 GPU (g2-standard-8). Data: WOD-E2E validation, the same
479 rater frames. Scored by Waymo's official RFS, analyzed by this repo's tested statistics.

## The question this settles

P2g's stability study showed that *frozen-encoder* perception grounding does not robustly
improve the label-free signal's prediction of human RFS, and named the one thing it could
**not** rule out: a **jointly-trained** end-to-end vision planner, whose ensemble
disagreement reflects *learned driving uncertainty* rather than frozen ImageNet/LVD
features. P2h runs exactly that, on a real GPU.

## Method

- **K = 6 end-to-end planners**, each a DINOv2-S backbone **fine-tuned jointly** with an
  ego+trajectory head (front camera + ego status → 20-waypoint future), trained on the
  WOD-E2E frames (bootstrap resample + distinct seed per member for ensemble diversity).
  This is a genuine driving-trained representation, not a frozen feature extractor.
- Each member predicts trajectories for the 479 held-out rater frames. We measure ensemble
  **disagreement** (mean pairwise displacement) vs the official **RFS**, with a drive-cluster
  bootstrap CI, and — applying the P2g lesson — a **member-bootstrap stability** distribution
  over the 15 four-of-six sub-ensembles (no single-draw claims).

## Result

479 rater frames / 93 drives. Min detectable |ρ| at this n = 0.128.

| signal | ρ (vs neg-RFS) | 95% CI (cluster) |
|---|---|---|
| **ensemble disagreement** (jointly trained) | **+0.202** | [+0.123, +0.280] |
| ADE-to-human (oracle anchor) | +0.458 | [+0.378, +0.529] |

Member-bootstrap stability (15 sub-ensembles): mean ρ = **+0.193**, sd 0.027,
range [+0.149, +0.234].

The same label-free signal, across every representation we tried on the human benchmark:

| representation | ρ (disagreement vs human RFS) |
|---|---|
| ego-status only (frozen) | ~0.18 (sd 0.05) |
| frozen DINOv2, front camera | ~0.16 |
| frozen DINOv2, 8-camera surround | ~0.12 |
| **jointly-trained vision (this study)** | **~0.19 (sd 0.03)** |

## Honest reading

1. **Joint training did not break through.** A real driving-trained vision ensemble lands at
   ρ ≈ 0.19–0.20 — the CI's upper bound (0.280) does not even reach the pre-registered 0.30
   bar, and it overlaps the blind ego baseline (~0.18). **H1 is not met, now confirmed even
   with the strongest version of the signal.**
2. **Two things *did* improve — neither is the thing that matters.** Joint training made the
   signal **tighter** (sd 0.027 vs ego's 0.05) and produced a genuinely **better planner**
   (ADE-vs-RFS rose to 0.458, the highest in the project — the vision planner predicts the
   human trajectory more accurately). But a better planner with a more stable disagreement
   signal *still* doesn't make that disagreement predict the human *rating* well.
3. **The real finding — a ceiling, not a tuning problem.** Across ego-only, frozen-image,
   surround, and now jointly-trained vision, label-free ensemble disagreement predicts human
   RFS at ρ ≈ 0.12–0.20 and never approaches 0.30. The ceiling does not move with perception,
   coverage, or end-to-end training. Ensemble disagreement and fine-grained human-rated
   driving quality are only loosely coupled — disagreement is a strong proxy for *open-loop
   error* (P2a, 0.70) and *binary safety events* (P2c, AUROC ~0.8), but a weak proxy for the
   nuanced quality the human raters score.

## Where it sits in the arc

| target | label-free disagreement |
|---|---|
| open-loop ADE (P2a) | strong — ρ = 0.70 |
| closed-loop PDMS score (P2b) | null — ρ ≈ 0 |
| closed-loop gate events (P2c) | decisive — AUROC ~0.8 |
| human RFS, frozen reps (P2e–g) | weak — ρ ≈ 0.12–0.18 |
| **human RFS, jointly-trained vision (P2h)** | **weak — ρ ≈ 0.19, does not clear 0.30** |

The headline result of the project is unchanged and now well-bounded: the cheap signal is
**decisive for the binary safety gates** and **weak for fine-grained human ratings**, and the
latter is a genuine ceiling — not fixed by better perception or by end-to-end training.

## What remains open (honestly, smaller now)

Not ruled out: a *much* larger driving foundation model, multi-camera + LiDAR + temporal
context, or a *learned* failure predictor trained directly on RFS (supervised, not
label-free). But the label-free, training-free framing — the cheap-signal thesis — is now
thoroughly tested against human ratings and is weak. That is the honest result.

## Reproduce

```bash
# parse pixels (wod env, mounted data):  python experiments/wod_e2e_rfs/run_parse_pixels.py
# train K planners (GPU torch env):       python experiments/wod_e2e_rfs/train_vision.py
# RFS eval (wod env):                     python experiments/wod_e2e_rfs/rfs_eval.py
# analyze (repo .venv):                   python experiments/wod_e2e_rfs/analyze_p2h.py
```

No Waymo frames redistributed — only per-frame derived scalars and the trained members'
predicted trajectories.
