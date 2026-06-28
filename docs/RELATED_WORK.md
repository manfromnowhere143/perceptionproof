# Related Work — what predicts closed-loop driving failure

A short, cited map of the 2024–2026 evidence that frames PerceptionProof's experiments. Every
number here is from a primary source and was adversarially fact-checked. It explains the P2b null
and motivates the P2c design.

## Open-loop metrics do not predict closed-loop safety

- Displacement error (L2/ADE/FDE) vs closed-loop Bench2Drive Driving Score: Spearman **ρ = −0.36,
  p = 0.43** — no correlation (arXiv 2605.00066). NAVSIM's own paper shows a blind ego-kinematics
  MLP attains SOTA displacement yet drives dangerously (arXiv 2406.15349, NeurIPS 2024).
- NAVSIM aggregate **PDMS** vs Driving Score: **ρ = 0.90** — far better than displacement, but
  non-monotonic, and dominated by Ego Progress (ρ = 0.83) rather than No-Collision (ρ = 0.45).

## Why: the PDMS gate structure

PDMS = (NC × DAC) × (5·EP + 5·TTC + 2·C) / 12 (NAVSIM `docs/metrics.md`, arXiv 2406.15349). NC
(no-at-fault collision) and DAC (drivable-area compliance) are **hard multiplicative gates** — a
single binary catastrophic event zeroes the whole score. Cross-scene PDMS variance is therefore
dominated by rare binary events, so a signal tracking smooth trajectory distance/spread cannot
track PDMS. This mechanistically explains why ensemble disagreement (open-loop ρ ≈ 0.70) fails to
transfer (closed-loop ρ ≈ 0).

## Pure uncertainty/disagreement fails closed-loop; collision-aligned signals transfer

- MC-Dropout **AUROC 50.1%**, GMM **49.4%** (chance) at predicting closed-loop collisions on
  NeuroNCAP (CATPlan, arXiv 2503.07425). This independently reproduces our P2b null.
- Supervised introspection (RiskMonitor, reads planner internal state, trained on collision loss):
  **AUROC 70.6%**, and gating a braking policy cuts collisions **69.9% → 23.0%** (arXiv 2503.07425).
- Label-free latent world-model trajectory-consistency (World4Drive, ICCV 2025): **NavSim PDMS
  85.1** with no perception labels (arXiv 2507.00603).
- Disengagement prediction from real drives (late state+camera fusion): >85% at 7 s lead, 20% FPR
  (Kuhn et al., IEEE T-ITS, doc 9310689).

## The open gap

Occupancy/collision-risk methods that *propose* the right signals — occupancy entropy, free-space
confidence, TTC/RSS surrogates — are validated **open-loop only**: Drive-OccWorld plans open-loop
on nuScenes (arXiv 2408.14197); RiskNet on logged trajectories (arXiv 2504.15541). **No published
work correlates a label-free occupancy/collision signal with per-scene NAVSIM PDMS.** That is the
gap PerceptionProof P2c targets.

## RFS (WOD-E2E) is open-loop

The Waymo Rater Feedback Score is a human-preference metric superior to ADE, but still open-loop;
its authors name closed-loop validation as future work (arXiv 2510.26125; Poutine, arXiv 2506.11234).

## Implication for PerceptionProof

Predict the **binary gate events** (NC=0, DAC=0) and the **TTC** sub-score with a **collision-aligned,
label-free** signal (trajectory–agent proximity / time-to-collision from the simulator's own agent
tracks and drivable area), not trajectory spread — and evaluate by AUROC on a real planner over a
larger, higher-power scene set, against the displacement-disagreement baseline. See
`PREREGISTRATION.md` (P2c) and `experiments/`.
