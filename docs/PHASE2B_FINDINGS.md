# Phase 2B — Findings: the closed-loop validity bridge via cheap features is a no-go

Date: 2026-06-29. Compute: GCP n2-standard-16, CPU. Data: NAVSIM / OpenScene, 760 scenes across
33 drives, **per-trajectory** closed-loop PDMS (real PDM simulator). The second remaining shot at
a positive breakthrough — where supervision is abundant (the simulator labels at scale, so the
Phase-2A data wall does not apply) — tested per the pre-registration's go/no-go.

## What we tested

The survey's open problem: recover the closed-loop ranking of trajectories with a *cheap* learned
predictor where the open-loop metric fails. We scored **per-trajectory** closed-loop PDMS for a
diverse set of trajectories per scene (6 ego-MLP ensemble members + constant-velocity + stopped
heuristics — ~8 per scene, 6,080 simulations), recorded each scene's human future, and asked:
does a cheap learned predictor of PDMS (trajectory geometry only, no simulator) recover the
closed-loop ranking **better than the open-loop ADE-to-human metric** the field uses? Held out by
drive; the decisive metric is the within-scene Kendall-τ (what an evaluation layer needs), with a
paired drive-cluster bootstrap.

## Result — no decisive win (no-go), stable across power levels

| | learned (cheap geometry, no sim) | ADE-to-human (open-loop) |
|---|---|---|
| overall held-out Spearman vs PDMS | +0.144 | **+0.404** |
| within-scene Kendall-τ vs PDMS | +0.083 | +0.060 |

Paired within-scene difference (learned − ADE): **+0.023, 95% CI [−0.040, +0.100] — inconclusive.**
The paired CI straddled zero at every checkpoint (280, 520, 680, 760 scenes); overall, the
open-loop ADE was consistently as-good-or-better. There is **no decisive win** for the cheap
learned predictor — the pre-registered bar for "go" was a CI excluding zero, and it is not met.

## Why — and why it unifies the whole program

Both signals are *weak* at predicting closed-loop PDMS because **closed-loop quality is
scene-interactive**: PDMS depends on how a trajectory threads *this* scene's agents and drivable
area, which trajectory *geometry alone* cannot see. The open-loop ADE does surprisingly okay only
because its reference — the recorded human trajectory — is itself a good closed-loop trajectory, so
"close to the human path" loosely tracks "safe." Neither cheap, scene-light signal recovers the
ranking well, and the learned one does not beat the metric.

This is the same wall, from the third direction. Across the program:

- **Cheap signals recover the binary safety boundary.** Label-free disagreement and
  collision-geometry predict the NC/DAC **gate events** at AUROC ~0.8 (Phase 1, P2c) — those events
  are a relatively scene-light, threshold-crossing property.
- **Cheap signals do not recover graded, scene-interactive quality.** Against human RFS (Phase 2A)
  a data wall blocks it; against closed-loop PDMS *ranking* (here) cheap geometry does not beat the
  open-loop metric. Graded quality lives in scene interaction that cheap, scene-light signals miss.

That is the honest, unifying thesis of the whole study, now triangulated from open-loop error,
closed-loop gates, closed-loop score-ranking, and human ratings: **cheap evaluation works for the
binary safety boundary an evaluation layer most needs, and provably does not substitute for
simulation or human raters on graded, scene-interactive quality — which is exactly why the field
needs them.**

## Reproduce

```bash
# CPU; reads the committed per-trajectory PDMS data (our trajectories + simulator PDMS + a
# precomputed ADE scalar; no ground-truth trajectory or frames redistributed).
python experiments/phase2_navsim_bridge/analyze.py experiments/phase2_navsim_bridge/bridge_data.json
```

`score_bridge.py` is the per-trajectory PDM scorer (run on a NAVSIM VM); `bridge_data.json` is its
license-clean derived output (6,080 scored trajectories); `verdict.json` the computed statistics.
