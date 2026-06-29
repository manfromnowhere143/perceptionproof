# Phase 2A — Findings: the supervised RFS target is data-walled (a pre-registered null with a mechanism)

Date: 2026-06-29. Compute: CPU (no GPU needed). Data: WOD-E2E validation, 479 rater-labeled
scenes. This is the honest interim verdict on the [Phase 2 program](PHASE2_PROGRAM.md)'s locked
target — reported per the pre-registration's rule that *a null at any gate is reported, not
buried*.

## What we tested

Phase 2's gates G1 (predict held-out RFS, beat the ADE baseline) and G3 (use the predictor as a
reward model to lift a planner's actual RFS). All on real planner trajectories, held out by scene.

## Results

**1. A synthetic-candidate probe looked strong — and we distrusted it correctly.**
Predicting RFS from a candidate's geometry+ego over *perturbed* trajectories gave held-out
Spearman **0.736** vs an ADE-to-human oracle's 0.39. But this was partly a perturbation artifact:
a model trained on perturbations transfers to *real* trajectories at only **0.262** (de-risk test B).

**2. On real planner trajectories, the reference-free predictor matches the oracle but does not beat it.**
A 6-member ego-MLP ensemble + driving heuristics gave genuine, non-perturbation candidates:

| held-out (real trajectories) | learned (geom+ego, no references) | ADE-to-human oracle (uses the label) |
|---|---|---|
| per-trajectory Spearman vs RFS | **0.573** | 0.603 |
| within-scene ranking | 0.523 | 0.509 |

The reference-free model essentially *ties* a label-using oracle — but **misses the G1 bar (0.60)** and does not beat the baseline.

**3. Frozen scene perception does not help.** Adding cached DINOv2 8-camera embeddings moved
per-trajectory Spearman 0.573 → 0.539, Δ = −0.034, 95% CI **[−0.100, +0.036]** — inconclusive
(Phase 1's ceiling, again).

**4. G3 fails: the reward model does not lift a planner's RFS.** Reranking a planner's plausible
proposals on held-out scenes:

| selection strategy | mean true RFS |
|---|---|
| random | 7.281 |
| planner default (ensemble mean) | **7.480** |
| our reward-model pick | 7.403 |
| oracle (best in pool) | 8.134 |

reranked − default = **−0.077, CI [−0.266, +0.112]** (inconclusive, slightly worse); only
**14%** of the random→oracle headroom captured.

## Why — the wall is fundamental, not a method failure

The decisive insight: **the only abundant supervision is the displacement signal.** Every one of
the 11,160 frames has a recorded human future, so training at scale only teaches "be close to the
human future" — i.e., it reproduces the ADE oracle. The structure that makes the Rater Feedback
Score *different* from displacement — multiple human-preferred trajectories, velocity-scaled trust
regions, multimodality — exists **only on the 479 rated frames** (RFS needs the human references,
and the test set is hidden). Therefore a supervised method **cannot learn
human-preference-beyond-displacement at scale on public data.** This is almost certainly *why* no
supervised RFS predictor exists, and our pre-registered tests confirm it: the reference-free model
reaches the displacement oracle and no further; a reward-model lift over a strong ensemble baseline
is blocked.

This is a genuine, citable result — a characterized data wall in human-preference learning for
driving — but it is **not** the breakthrough Phase 2 set out to find. We report it as the honest
outcome.

## Consequence for the campaign

The human-RFS target is data-walled for a supervised win on public data. The campaign's remaining
path to a *positive* breakthrough is the target where supervision is **abundant and computable** —
the **closed-loop** validity bridge on NAVSIM, where the PDM simulator generates labels at scale
for as many trajectories/planners/scenes as we run (Phase 1 already scored 1,317 scenes). There the
data wall does not apply. Whether that target is winnable (vs the Pseudo-Simulation bar, r=0.89) is
the next strategic question — a decision, not an execution detail.

## Reproduce

```bash
# CPU; against the WOD-E2E-derived cache (rater references + ego features). No frames redistributed.
python experiments/phase2_rfs/probe_g1.py          # synthetic-candidate probe (inflated 0.736)
python experiments/phase2_rfs/derisk_g1.py         # real trajectories: 0.573 vs oracle 0.603; cross-dist 0.262
python experiments/phase2_rfs/derisk_g1_scene.py   # frozen scene perception: no help (Δ CI straddles 0)
python experiments/phase2_rfs/g3_rerank.py         # reward-model reranking: no lift (Δ=-0.077)
```

Derived scalar outputs are committed alongside (`*_out.json`).
