# PerceptionProof Phase 2 — A Learned Human-Preference Model for Driving

**Pre-registration (frozen before results). Campaign target, bar, method, and honest risk.**

This document is written *before* any Phase-2 result exists, so the bar cannot move after the
fact. It is the scientific successor to Phase 1, which established — with published nulls — that
*label-free* signals predict open-loop error (ρ=0.70) and binary safety gates (AUROC ~0.8) but
hit a **ceiling against human ratings** (ρ≈0.18, did not move with perception or end-to-end
training). Phase 2 asks the obvious next question Phase 1 earned the right to ask: **can a
*supervised* model break that ceiling — and is it the first of its kind?**

## 1. What the frontier actually says (verified, cited)

From a 103-agent adversarially-verified survey (2024–2026 literature; claims below survived 3-vote
verification unless noted):

- **The metric the field uses to rank planners is broken, and this is quantified.** Displacement
  metrics (L2/ADE/FDE) do *not* predict closed-loop driving: Spearman ρ = **−0.36** (p=0.43, n=7)
  vs Bench2Drive Driving Score; an MLP on ego-state alone reaches SOTA open-loop L2 (AD-MLP;
  "Is Ego Status All You Need", CVPR 2024). [arXiv 2605.00066, 2305.10430, 2312.03031]
- **Within Waymo's human benchmark, better ADE does not mean better human rating.** RFS
  (Rater Feedback Score) is a *non-learned, geometric* metric; better ADE ≠ better RFS (named
  inversions: WayNet best ADE but low RFS; HMVLM worse ADE but top RFS). [arXiv 2510.26125]
- **There is no supervised RFS / human-preference predictor in the literature.** Verified
  whitespace. The WOD-E2E leaderboard is tightly clustered (NaiveEMMA 7.528 → Poutine 7.986;
  expert ground-truth ceiling ≈ 8.13). [arXiv 2510.26125, 2506.11234]
- **Failure-prediction primitives exist but none predicts human/closed-loop planner quality.**
  QAD (task-relevant prediction failures, AUROC 0.946), the regret metric (CoRL 2024), RiskMonitor
  (collision-risk from planner tokens, AUROC 92.2% / AP 43.2% open-loop). All narrow; none is a
  learned model of *human driving preference*. [arXiv 2207.12380, 2403.04745, 2503.07425]
- **The cheap-evaluator bar to beat (alternative target) is Pseudo-Simulation**: r=0.89, R²=0.8
  alignment with closed-loop across 83 planners (prior best R²=0.7). [arXiv 2506.04218]

**Honest caveats the survey flagged (these shape the design):** the open-loop↔closed-loop
ranking-inversion evidence rests on only n≈7–8 paired planners from heterogeneous self-reported
papers — *statistically fragile*; the companion claim that PDMS "strongly recovers" the ranking
(ρ=0.90) was **refuted 0-3**. RFS margins are narrow. So any result that depends on *ranking ~10
planners* is underpowered by construction — the design must put its primary statistics where n is
large (per-trajectory), not where n is tiny (per-planner).

## 2. The locked target (one sentence)

> **Build the first supervised, calibrated model of human driving preference — a learned RFS /
> trajectory-quality predictor — and show it ranks trajectories by human preference, and lifts a
> planner's actual human rating, where displacement metrics provably fail.**

Why this one, over beating Pseudo-Simulation:
1. **It is verified open whitespace** — no supervised RFS predictor exists. Beating Pseudo-Sim is
   incremental (R²=0.8→?) on turf owned by the NAVSIM group with home-field resources.
2. **It is the honest continuation of our published null.** "Label-free hit a ceiling on human
   ratings, so we built the supervised model that breaks it" is a story a referee respects.
3. **It plays to our exact arsenal** — we already have the official RFS scorer, the WOD-E2E data
   (479 rater frames + 93 shards), a GPU training loop, and the tested ranking statistics
   (Spearman/Kendall-inversion/cluster-bootstrap). Our asymmetric advantage is the *predictor /
   validity layer*, not raw planning — and this is exactly that layer.
4. **It puts the primary statistics where n is large** (per-trajectory quality prediction over
   thousands of scored trajectories), side-stepping the fragile small-planner-count ranking.

## 3. The bar (pre-registered, falsifiable)

A win requires **both** core gates; the leap is a stretch gate.

| Gate | Metric | Bar to clear |
|---|---|---|
| **G1 — predicts human preference** | per-trajectory: predicted-RFS vs true-RFS on **held-out scenes**, drive-cluster bootstrap | Spearman ρ ≥ 0.6, CI excludes the ADE/L2 baseline (which is the published failing comparator) |
| **G2 — recovers preference where displacement fails** | rank trajectories within scene by predicted quality vs by ADE; agreement with human RFS ranking (Kendall-τ) | learned-model τ − ADE τ > 0, paired bootstrap CI excludes 0 |
| **G3 (stretch / the impressive leap)** | use the model as a re-ranker/reward over a base planner's candidate trajectories; measure the base planner's **actual RFS** on held-out WOD-E2E | RFS lift over the un-re-ranked planner, CI excludes 0 |

G1+G2 = a genuine, novel, defensible contribution (first learned human-preference model for
driving, beating the metric the field admits is broken). G3 = the result that makes experts sit
up: a human-preference *reward model* that measurably improves a driver — the driving analogue of
RLHF reward modeling.

## 4. Method hypothesis

A model `q(scene, trajectory) → predicted human rating`, trained on abundant supervision: on every
training scene, *any* candidate trajectory's RFS is computable geometrically from the human
reference trajectories — so we can generate millions of (scene, trajectory, RFS) pairs (real
planner outputs + structured perturbations) and learn the human-preference surface *without* the
references at test time. Backbone: a frozen or lightly-tuned scene encoder (we have the DINOv2
pipeline) + trajectory encoder + calibrated regression head. The novelty is not the architecture;
it is **the task and its validation** — the first learned surrogate of human driving preference,
held to ranking-recovery against the human ground truth.

## 5. Staged plan with go/no-go gates

- **Phase A — feasibility + G1.** Build the (scene, trajectory) → RFS dataset from WOD-E2E; train
  the predictor; evaluate held-out per-trajectory prediction vs the ADE baseline. **Gate: G1.** If
  the learned model cannot beat ADE on held-out RFS prediction with a CI excluding the baseline,
  we stop and report it honestly (a second clean null — still publishable).
- **Phase B — G2 + ablations.** Ranking recovery vs ADE; calibration; ablate scene vs trajectory
  features; robustness across the long-tail scenario types. **Gate: G2.**
- **Phase C — G3, the leap.** Re-rank a base planner's candidates with the learned reward; measure
  real RFS lift on held-out scenes; if it holds, prepare a leaderboard submission.
- **Phase D — write-up + (optional) leaderboard.** Pre-registered, cluster-bootstrapped, ablated,
  reproducible-from-committed-data, adversarially reviewed — the Phase-1 standard.

Each gate is objective and pre-registered. No phase advances on hope. A null at any gate is
reported, not buried.

## 6. Honest risk read

- **Narrow RFS ceiling (8.13).** A G3 leaderboard lift will be *small in absolute terms* even if
  real. We frame the contribution as the **predictor + ranking-recovery** (G1/G2, large-n, solid),
  with G3 as corroboration — not as "we topped the leaderboard by a lot."
- **Reward-hacking risk in G3.** A learned reward optimized against can be gamed; we guard with
  held-out scenes, the *geometric* RFS as the final judge (not our own model), and ablations.
- **Generalization.** Predicting RFS on held-out *scenarios* (not just held-out frames of seen
  scenes) is the real test; we split by scenario/segment, not by frame.
- **It may null.** The honest possibility is that human preference on the long tail is too
  irreducible to learn from this data at this scale. If so, that is itself a strong, citable
  finding ("even supervised, the human-rating surface resists cheap prediction") — and Phase 1 +
  Phase 2 together become a complete, honest map of what is and isn't learnable here.

## 7. Resources

Public data only (WOD-E2E, already on disk), single-digit GPUs (L4/A100 quota confirmed), our
existing scorer + stats + training loop. No fleet data, no proprietary anything. Feasible.

---

*Frozen: 2026-06-29. Survey: 103-agent adversarial verification, 21 primary sources, 23/25 claims
confirmed. This pre-registration may be refined for clarity but its gates (Section 3) will not be
weakened after results exist.*
