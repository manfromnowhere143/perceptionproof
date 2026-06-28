# P2c — Do label-free signals predict closed-loop PDMS *gate events*?

Date: 2026-06-28. Compute: GCP n2-standard-16, no GPU. Real NAVSIM PDM simulator, parallel scoring.

## The question, sharpened by the literature

P2b found ensemble disagreement does **not** predict the aggregate closed-loop PDM score
(ρ ≈ 0). The verified literature (`docs/RELATED_WORK.md`) explained why: PDMS is gated by binary
multipliers — **NC** (no-at-fault collision) and **DAC** (drivable-area compliance) — that zero
the whole score on a single catastrophic event, so a signal tracking the smooth score cannot
track it; the fix is to **predict the binary gate events directly**. P2c tests exactly that, and
asks a second question: does a **collision-aligned** geometric signal beat the displacement
baseline on those gates?

## Setup (real closed-loop, properly powered)

- **1,317 real NAVSIM scenes across 55 driving logs** (drive-stratified for tight cluster CIs),
  each with a precomputed PDM metric cache. Parallel PDM scoring, 0 errors.
- Deployed planner: a native ego-status MLP (member 0 of an ensemble).
- Targets (from the PDM scorer): **NC gate** (collision, 51 events), **DAC gate** (off-road, 94
  events), any-gate (PDMS=0, 349).
- Signals (label-free, no sensors — computed from the metric cache's own agent tracks + drivable
  map): **collision_risk** (min-distance ego→agents), **off_road** (ego-footprint fraction outside
  the drivable area), and the **disagreement** baseline (ensemble trajectory spread).
- Stats: AUROC with drive-clustered bootstrap CIs, and a **paired AUROC-difference** test
  (bootstrap by drive) for the decisive signal-vs-baseline comparison. Code:
  `experiments/navsim_p2c/`.

## Result

| Signal → gate | AUROC | 95% CI |
|---|---|---|
| collision_risk → **NC** (collision) | 0.772 | [0.678, 0.849] |
| disagreement → **NC** | 0.828 | [0.782, 0.871] |
| off_road → **DAC** (off-road) | 0.810 | [0.763, 0.855] |
| disagreement → **DAC** | 0.772 | [0.731, 0.817] |
| any signal → any-gate (PDMS=0) | 0.57–0.62 | — |

**Clean verdict — paired AUROC difference (matched − baseline), bootstrap by drive:**

| Comparison | Δ AUROC | 95% CI | p | verdict |
|---|---|---|---|---|
| collision_risk − disagreement → NC | −0.057 | [−0.147, +0.017] | 0.14 | inconclusive |
| off_road − disagreement → DAC | +0.039 | [−0.025, +0.100] | 0.21 | inconclusive |

## What it means (two findings, both honest)

1. **DECISIVE — gate reframing works.** Label-free signals predict the binary safety-gate events
   at **AUROC 0.77–0.83**, with CIs excluding chance, across 55 drives — for *both* the collision
   (NC) and off-road (DAC) gates. The same disagreement signal was at chance against the aggregate
   PDMS (P2b). **Reframing the target from the smooth score to the binary gates is what unlocks
   closed-loop predictability.** No prior published work had shown a per-scene label-free signal
   predicting NAVSIM PDMS gate events — this fills that gap.

2. **HONEST NULL — no signal dominates.** Neither the collision-geometry signal nor ensemble
   disagreement is the decisive winner on its matched gate; every paired difference includes zero.
   The *target reframing* matters; the *specific signal choice* does not, at this power. The
   mechanistic hypothesis ("collision-aligned geometry beats displacement-disagreement") is **not
   confirmed**.

## The arc (P2a → P2b → P2c)

| Phase | Outcome | Result |
|---|---|---|
| P2a | open-loop trajectory error (ADE) | disagreement predicts it, ρ = 0.70 (leave-one-out 0.68) |
| P2b | closed-loop PDMS *score* | disagreement does **not**, ρ ≈ 0 (matches MC-Dropout AUROC 50% in the literature) |
| P2c | closed-loop PDMS *gate events* | label-free signals **do**, AUROC ~0.8 |

Open-loop predictability does not transfer to the closed-loop *score*, but label-free signals do
predict closed-loop *failure events* when the target is the binary safety gates directly.

## Robustness: leave-one-out NC (coupling removed)

The deployed planner (member 0, whose NC gate is the outcome) is also inside the disagreement
ensemble — a potential coupling. Removing it: disagreement among the **other** members {1,2,3}
predicts member 0's collision at **AUROC 0.821 [0.775, 0.865]**, essentially identical to the
coupled 0.829 and with the CI excluding chance. So the disagreement→collision signal is genuine
scene difficulty, not an algebraic artifact. Reproduce: `experiments/navsim_p2c/leave_one_out_nc.py`.

## Honest caveats

1. **Weak deployed planner** (ego-status MLP; its failures are partly non-collision gates like
   driving-direction), which both bounds and explains the any-gate noise. A real sensor planner
   (TransFuser) whose failures are genuine collisions is the natural P2d step.
2. ~~Deployed-planner-in-ensemble coupling~~ — ADDRESSED by the leave-one-out test above.
3. **Power**: 51 collision / 94 off-road events across 55 drives give clear above-chance verdicts
   but leave the *signal-superiority* test underpowered (Δ≈0.04–0.06).

## Reproduce

`experiments/navsim_p2c/` — drive-stratified token generation, parallel PDM scoring with the
collision-geometry signal, and the paired-difference analysis. No OpenScene/nuPlan data
redistributed; aggregate numbers only (`DATA_LICENSES.md`).
