# What Label-Free Ensemble Disagreement Predicts in Driving Evaluation: From Open-Loop Error to Human Ratings

**A pre-registered study with published nulls and a self-correction**

PerceptionProof project — open code and signed receipts: <https://github.com/manfromnowhere143/perceptionproof>

---

## Abstract

Autonomous-driving research openly concedes that its evaluation does not predict safety: a
2026 cross-benchmark study found open-loop planning metrics *mis-rank* closed-loop driving,
with explicit ranking inversions, and the field's current substitute for ground truth is
expensive human raters (Waymo's Rater Feedback Score on long-tail segments). We ask a narrow,
falsifiable question: **how far does a single cheap, label-free signal — ensemble trajectory
disagreement — go toward predicting failure as the prediction target moves from open-loop
error, to the closed-loop driving score, to the binary closed-loop safety events, to human
ratings?** We do not claim that disagreement is a novel uncertainty estimate (it is not;
Deep Ensembles, 2017). Our contribution is the *bridge*: one signal, traced across four
targets of increasing realism on two real benchmarks (NAVSIM/nuPlan and Waymo WOD-E2E) under
one pre-registered protocol, with drive-level cluster-bootstrap confidence intervals and
tamper-evident provenance. The result has a clear shape. The signal predicts open-loop error
strongly (Spearman ρ = 0.70), does **not** transfer to the smooth closed-loop PDM score
(ρ ≈ 0), but **decisively** predicts the binary closed-loop safety gates — collision and
off-road — that the score is built on (AUROC ≈ 0.8). Against human ratings it is **weak**
(ρ ≈ 0.18) and stays weak: a 20-seed-set stability study overturns a preliminary positive of
our own, and a GPU-trained end-to-end vision ensemble — the strongest version of the signal —
reaches only ρ = 0.20 (95% CI upper bound 0.28), failing to clear our pre-registered 0.30
threshold. The weakness against human ratings is a **ceiling**, not a tuning problem: it does
not move with perception, camera coverage, or end-to-end training. We report every null and
the self-correction in full, because the integrity of the evaluation is the contribution.

---

## 1. Introduction

The central, unglamorous problem of autonomous-driving research is not perception or planning
in isolation but **evaluation that predicts safety**. Open-loop displacement metrics are cheap
and reproducible, but a 2026 cross-benchmark analysis demonstrated that they mis-rank
closed-loop safety, producing clear ranking inversions [RI2026]. Closed-loop simulation is
better but expensive, and on the long tail the field increasingly relies on human raters:
Waymo's End-to-End driving benchmark (WOD-E2E) ships a Rater Feedback Score (RFS) that grades
a proposed trajectory against human-preferred trajectories inside a speed-scaled trust region
[RFS2026].

A natural hope is that *cheap, label-free* signals — quantities computable at inference time
with no ground-truth future and no human labels — could triage failures and recover the
human-judged ranking at a fraction of the cost. The oldest such signal is ensemble
disagreement: train several models, and treat the spread of their predictions as an
uncertainty estimate [DE2017]. We emphasize at the outset that **we claim no novelty for the
signal itself.** Disagreement-as-uncertainty is two decades old.

What is, to our knowledge, not done carefully is the *bridging study*: taking one such signal
and measuring, on real data under one frozen protocol, exactly which evaluation targets it
predicts and which it does not — open-loop error, the closed-loop score, the closed-loop
*events*, and human ratings — with honest confidence intervals and pre-registered thresholds.
That is this paper. Our contributions are:

1. **A single label-free signal traced across four targets of increasing realism** on two real
   benchmarks (NAVSIM/nuPlan; Waymo WOD-E2E), under one pre-registered analysis.
2. **A clean empirical shape with three decisive findings and two informative nulls**: strong
   on open-loop error; a reproduced open-loop-to-closed-loop *gap* on the smooth score;
   decisive on the binary closed-loop safety gates; weak on human ratings.
3. **A demonstrated ceiling on human-rating prediction**: the weak human-RFS correlation does
   not improve with frozen perception features, full surround-camera coverage, or — tested on
   a GPU — a jointly-trained end-to-end vision ensemble.
4. **A worked example of evaluation integrity**: a preliminary positive result of ours did not
   survive a stability study and is retracted in-line; we argue for reporting distributions,
   not single draws, for small effects.
5. **Tamper-evident provenance**: every reported number is produced by unit-tested statistics
   and an Ed25519 hash-chained receipt layer; no dataset frames are redistributed.

A null on the cheap-signal hope is a real, useful finding — it tells the field where label-free
triage works (safety events) and where it does not (fine-grained human quality). We report it
as such.

---

## 2. Related Work

**Uncertainty and disagreement.** Deep Ensembles [DE2017] established ensemble disagreement as a
predictive-uncertainty estimate; a large literature on selective prediction and failure
detection follows (e.g., risk-coverage analysis [GEY2017]). Our signal S1 is a trajectory-space
instance of this idea and is deliberately not novel.

**Open-loop vs closed-loop evaluation.** NAVSIM [NAVSIM2024] introduced the non-reactive PDM
score (PDMS), a principled closed-loop-style metric that combines a no-at-fault-collision gate
(NC), a drivable-area-compliance gate (DAC), time-to-collision (TTC), ego progress (EP), and
comfort (C): PDMS = (NC × DAC) × (5·EP + 5·TTC + 2·C)/12. The gates are binary multiplicative.
The 2026 ranking-inversion study [RI2026] showed that open-loop metrics fail to predict
closed-loop safety. We reproduce the *gap* with our own signal and then localize where the
signal does work inside PDMS.

**Human-rated long-tail evaluation.** WOD-E2E [RFS2026] provides per-frame rater-preferred
trajectories with quality labels and an official scorer; it is, at present, the closest thing
to ground-truth driving quality on the long tail. We treat RFS as the hardest and most
decision-relevant target.

**Multiple comparisons and clustered data.** We control the false-discovery rate with
Benjamini-Hochberg [BH1995] and compute all confidence intervals by cluster bootstrap at the
drive level so that correlated frames from one drive cannot inflate confidence.

---

## 3. Problem Setup and Definitions

### 3.1 Targets

For a planner (or planner ensemble) and a set of scenes, we relate a label-free signal to four
targets of increasing realism:

- **T1 — open-loop error.** Average displacement error (ADE) of the predicted trajectory
  against the recorded human future. Cheap, but known to mis-rank safety.
- **T2 — closed-loop score.** The smooth NAVSIM PDM score in [0, 1].
- **T3 — closed-loop gate events.** The *binary* safety gates the score is built on: NC
  (no at-fault collision) and DAC (drivable-area compliance). These are the events a safety
  monitor actually cares about.
- **T4 — human rating.** Waymo's Rater Feedback Score, the official human-judged trajectory
  quality on long-tail segments.

### 3.2 The label-free signal

Our primary signal is **S1, ensemble disagreement**: train K planners, and for each scene take
the mean pairwise displacement among the K predicted trajectories. It uses no ground-truth
future and no human label. The repository also implements and unit-tests a four-signal family
(S1 ensemble disagreement; S2 SE(2)-aligned temporal flicker; S3 corridor occupancy-conflict;
S4 VLA semantic-entropy); the headline empirics in this paper use S1, plus a label-free
collision-geometry signal in the gate-event experiment (Section 5.3). We are explicit about
this scope: the study's strength is the careful target-by-target measurement of one cheap
signal, not breadth of signals.

As an **oracle anchor** we also report ADE-to-human, which requires the recorded future and is
therefore *not* label-free; it certifies that a target is predictable at all from trajectory
quality, bounding what any cheap signal could hope to achieve.

### 3.3 Pre-registered hypotheses

Frozen before results (`PREREGISTRATION.md`):

- **H1** — a label-free signal predicts the per-segment target: Spearman ρ ≥ 0.30, BH-corrected
  q < 0.05.
- **H2** — a signal-adjusted ranking beats the open-loop metric (Kendall distance strictly
  lower; bootstrap CI > 0).
- **H3** — the signal triages failures better than chance (AP > base rate, E-AURC < random).

A null on all three is a reported finding. The 0.30 bar in H1 is the threshold we hold the
human-rating result to throughout.

---

## 4. Statistical Methodology

All correlations are Spearman ρ with a **drive-level cluster bootstrap** (10,000 resamples):
we resample drives (not scenes), so correlated scenes from one drive cannot inflate confidence,
and report the 2.5/97.5 percentile interval and a permutation p-value. For event targets we
report failure-mining AUROC / average precision / precision-at-k against the base rate, and
risk-coverage AURC / E-AURC for the selective-prediction view. When two signals compete on the
same target we use a **paired** bootstrap: resample drives once per iteration and recompute both
statistics on the same resample, so the difference isolates the signal. Across hypotheses we
apply Benjamini-Hochberg FDR control. We report the **minimum detectable** |ρ| at each sample
size (power 0.8) so that weak-but-real and underpowered are never confused.

Every experiment emits an Ed25519 hash-chained receipt; the chain verifies independently and
re-running is byte-identical. Datasets (NAVSIM/nuPlan, WOD-E2E) carry non-commercial research
licenses; we redistribute **no** frames — only segment ids and derived scalars.

---

## 5. Experiments and Results

All experiments use real data: NAVSIM/OpenScene scenes scored by the real PDM simulator
(Sections 5.1–5.4) and the WOD-E2E validation split scored by Waymo's official RFS
(Sections 5.5–5.7). Planners range from ego-status MLP ensembles to a 3-seed pretrained
camera+LiDAR TransFuser and a GPU-trained end-to-end vision ensemble.

### 5.1 T1, open-loop error: strong (P2a)

On 788 held-out NAVSIM scenes (disjoint by-log split; 4-member ego-status MLP ensemble),
label-free disagreement predicts open-loop ADE: **ρ = 0.699 [0.599, 0.750]**, p = 0.0005;
failure-mining AUROC = 0.855, AP = 0.846 (precision@50 = 0.98); E-AURC = 0.180. Because
disagreement and ADE share the same predictions, we ran a **leave-one-out** control: the
disagreement of members {0,1,2} against the error of held-out member 3 gives
**ρ = 0.683 [0.589, 0.729]** — retiring the structural-coupling concern. The cheap signal
genuinely tracks scene difficulty in the open-loop regime.

### 5.2 T2, closed-loop score: null (P2b)

On 400 NAVSIM scenes (8 drives) scored by the real PDM simulator, the same disagreement signal
does **not** transfer to the smooth PDM score: **ρ = −0.074 [−0.396, 0.285]** (cluster CI
includes zero), failure-mining AUROC ≈ 0.53 — versus the robust open-loop ρ = 0.70 on the same
signal. This is a pre-registered null, published unmodified. It reproduces, with our own signal,
the open-loop-to-closed-loop gap that motivates the field [RI2026].

### 5.3 T3, closed-loop gate events: decisive (P2c)

The smooth score hides the events. Reframing T2's target from the PDMS scalar to its **binary
gates** changes the picture entirely. On 1,317 NAVSIM scenes across 55 drives (real PDM
simulator, parallel scoring), label-free signals predict the NC (collision) and DAC (off-road)
gate events at **AUROC 0.77–0.83**, with confidence intervals that exclude chance — decisive.
A leave-one-out variant (disagreement of members {1,2,3} vs held-out member-0 collision) gives
**AUROC 0.821 [0.775, 0.865]**, ruling out coupling as the explanation.

We also report an **honest null within the win**: a label-free collision-geometry signal
(minimum distance + footprint off-road computed from the simulator's tracked objects and
drivable-area map, no sensors) versus ensemble disagreement are statistically *indistinguishable*
on their matched gates — every paired AUROC-difference CI includes zero. The lesson is that the
**reframing** (score → events) matters more than the **choice of signal**.

The arc so far — ρ = 0.70 open-loop, ρ ≈ 0 closed-loop score, AUROC ≈ 0.8 closed-loop gates —
is the project's headline: the cheap signal is decisive exactly where an evaluation layer needs
it, the binary safety events.

### 5.4 T3 with a real sensor planner: underpowered (P2d)

To test the gates with a real perception planner we ran a 3-seed pretrained **TransFuser**
(camera + LiDAR) end-to-end through the PDM simulator: 396 scenes, 52 drives, 0 errors. The
pipeline is validated, but the result is **underpowered, not refuted** — a strong planner almost
never fails (3 collisions, 12 off-road events on 396 scenes), too few to estimate a
failure-prediction AUROC; only the broad any-gate target is borderline (disagreement AUROC
0.600 [0.514, 0.697]). A powered real-sensor measurement needs a much larger frame-consistent
sensor dataset. We report this honestly rather than overstate a small-sample number.

### 5.5 T4, human ratings, ego-only: weak (P2e)

We now turn to the hardest target. On the WOD-E2E validation split we trained a 4-seed
ego-status MLP ensemble (ego velocity/acceleration + driving intent → 20-waypoint future) and
evaluated on **479 rater-labeled frames across 93 drives**. Minimum detectable |ρ| at this n is
0.128. Label-free disagreement predicts worse RFS at **ρ = 0.151 [0.063, 0.237]** (BH q < 0.05)
— statistically real but **below the pre-registered 0.30 bar**, so H1 is not met. The oracle
anchor (ADE, which needs the human label) reaches **ρ = 0.395 [0.326, 0.458]**: RFS *is*
predictable from trajectory quality; the cheap ego-only signal simply recovers little of it.

### 5.6 The self-correction: perception grounding does not robustly help (P2f → P2g)

A natural hypothesis is that the ego-only ensemble is *blind* to scene difficulty. We tested it
by adding a frozen DINOv2 front-camera embedding to the input (P2f). A single ensemble
instantiation suggested a lift — disagreement-vs-RFS rising from 0.10 (ego) to 0.22 (vision),
with a paired bootstrap whose CI cleared zero by +0.003. We initially reported this as a win.

It did not survive scrutiny. Pushing to the full 8-camera surround **inverted** the ranking
(ego 0.29 > front 0.16 > surround 0.10) in a single run, and the ego arm itself swung from 0.10
to 0.29 between runs — the signature of a correlation dominated, at this effect size, by which
seeds and training sample the ensemble drew. We therefore ran the decisive test: re-train each
representation over **20 independent seed-sets**, paired (same seeds across representations per
instantiation). The distribution of ρ:

| representation | mean ρ | sd | range |
|---|---|---|---|
| ego-status (frozen) | +0.176 | 0.048 | [0.075, 0.262] |
| DINOv2 front camera | +0.161 | 0.026 | [0.112, 0.218] |
| DINOv2 8-camera surround | +0.124 | 0.023 | [0.084, 0.168] |

Paired per-instantiation differences all straddle zero — front − ego [−0.092, +0.076],
surround − ego [−0.133, +0.033] — with P(vision > ego) ≈ 0.10–0.30, and point estimates that
favor *ego*. **Frozen-encoder perception grounding does not robustly improve the signal.** The
P2f lift was seed noise; we retract it in-line. The methodological point generalizes: for small
effects, report distributions over seed-sets, not single draws.

### 5.7 T4 with a jointly-trained vision ensemble: the ceiling (P2h)

The one thing the stability study could not rule out is that frozen ImageNet/LVD features are
the wrong representation, and that a *jointly-trained* end-to-end vision ensemble — whose
disagreement reflects learned driving uncertainty — would do better. We tested this directly on
a GPU: **K = 6 DINOv2-S planners fine-tuned end-to-end** (front camera + ego → trajectory), each
on a bootstrap of the training frames with a distinct seed, evaluated on the same 479 rater
frames.

| signal | ρ vs RFS | 95% CI (cluster) |
|---|---|---|
| ensemble disagreement (jointly trained) | **+0.202** | [+0.123, +0.280] |
| ADE-to-human (oracle anchor) | +0.458 | [+0.378, +0.529] |

Member-bootstrap stability over the 15 four-of-six sub-ensembles: mean ρ = +0.193, sd 0.027,
range [0.149, 0.234]. Joint training did two real things — it made the signal **tighter**
(sd 0.027 vs ego's 0.048) and produced a genuinely **better planner** (ADE-vs-RFS = 0.458, the
highest in the study) — but the disagreement-vs-RFS correlation's CI tops out at **0.280, below
the 0.30 bar**, and overlaps the blind ego baseline. H1 is not met even here.

---

## 6. Discussion

**The shape, summarized.** One label-free signal, four targets:

| target | label-free disagreement | reading |
|---|---|---|
| T1 open-loop ADE | ρ = 0.70 | strong |
| T2 closed-loop PDMS score | ρ ≈ 0 | null (reproduces the gap) |
| T3 closed-loop gate events | AUROC ≈ 0.8 | decisive |
| T4 human RFS (best of four representations) | ρ ≈ 0.20 | weak, below 0.30 |

The signal is most useful precisely where a safety-evaluation layer needs it — the binary
collision/off-road events — and least useful for the smooth score that the field already knows
is broken. That is an encouraging and actionable result for triage: cheap disagreement is a
viable pre-filter for closed-loop *safety events*.

**The human-rating ceiling.** Against fine-grained human ratings the signal is weak and, more
importantly, *capped*. Across ego-only, frozen front-camera, full surround, and a GPU-trained
end-to-end vision ensemble, the disagreement-vs-RFS correlation sits at ρ ≈ 0.12–0.20 and never
approaches 0.30. The ceiling does not move with perception, camera coverage, or end-to-end
training, even as the underlying planner demonstrably improves (ADE-vs-RFS rising from 0.40 to
0.46). The most parsimonious explanation is that ensemble disagreement and human-rated
*quality* are only loosely coupled: disagreement captures where models find a scene hard to
*predict*, which aligns with open-loop error and with whether a binary safety boundary is
crossed, but not with the graded, preference-shaped quality a human assigns to a trajectory.

**Why this is the useful answer.** The cheap-signal hope was that label-free triage could
substitute for human raters on the long tail. Our evidence says: not via ensemble disagreement,
and not because the planner is too weak — the limitation is intrinsic to what the signal
measures. A learned failure predictor trained *directly* on RFS (supervised, not label-free) is
the natural alternative, and is outside the cheap-signal framing this paper tests.

---

## 7. Limitations and Threats to Validity

- **Signal scope.** The headline empirics use S1 (disagreement) and a collision-geometry signal;
  S2–S4 are implemented and unit-tested but not run at scale on real data. We do not claim S1 is
  the best possible label-free signal.
- **Planner strength.** Several experiments use ego-status MLP planners; the real-sensor gate
  experiment (P2d) was underpowered because a strong planner rarely fails. The headline gate
  result (P2c) uses such an ego planner; the real-sensor confirmation remains open for lack of a
  large frame-consistent sensor dataset.
- **Single splits.** NAVSIM experiments use our own by-log splits, not the official navtest; the
  WOD-E2E result uses one validation split (479 rater frames). The cluster bootstrap controls
  per-drive correlation but cannot substitute for multiple datasets.
- **Vision-ensemble scope (P2h).** K = 6, front camera, DINOv2-S, four epochs; stability is
  estimated by member bootstrap (which can under-state variance) rather than many independent
  full ensembles. A much larger driving foundation model, multi-camera + LiDAR + temporal
  context, is not ruled out — but the *cheap, training-free* framing this paper tests is
  thoroughly negative against human ratings.
- **RFS as ground truth.** RFS is itself a human-rater construct with its own trust-region and
  labeling choices; we treat it as the target, not as perfect truth.

---

## 8. Reproducibility and Provenance

Signals, validity statistics, and the receipt chain are unit-tested (33 tests). **Every headline
number in this paper regenerates from committed derived data** — the scored per-scene outputs
(segment ids, predicted trajectories, gate flags, RFS values; no frames) are committed next to
each analysis, and the unit-tested statistics recompute the reported figures with no dataset
download and no GPU. For example, `python experiments/navsim_p2c/analyze.py
experiments/navsim_p2c/pp_p2c_scaled.json` reproduces the gate-event AUROCs and the paired-null
verdict; `python experiments/wod_e2e_rfs/analyze_p2h.py` reproduces the jointly-trained-vision
result. The full data-acquisition and scoring pipelines (which are dataset-bound) ship alongside,
under each `experiments/<name>/` (setup, data, train, score). A synthetic end-to-end run
exercises the full harness with no data or GPU and verifies the Ed25519 receipt chain.
Pre-registered hypotheses and thresholds are frozen in `PREREGISTRATION.md`. Datasets are under
non-commercial research licenses; this work redistributes none of their frames — only segment ids
and our derived outputs.

---

## 9. Conclusion

We traced one cheap, label-free signal — ensemble disagreement — across four driving-evaluation
targets of increasing realism, on real data, under one pre-registered protocol. The signal is
strong on open-loop error, null on the smooth closed-loop score, **decisive on the binary
closed-loop safety events**, and weak on human ratings — and the human-rating weakness is a
**ceiling** that does not yield to perception, camera coverage, or GPU-scale end-to-end
training. Along the way a preliminary positive of ours failed a stability test and was retracted.
The practical conclusion is specific and, we believe, useful: label-free ensemble disagreement
is a viable triage signal for closed-loop **safety events**, but not a substitute for human
raters on fine-grained long-tail **quality**. We publish the nulls, the self-correction, and the
signed receipts, because for an evaluation method the integrity is the result.

---

## References

- [DE2017] Lakshminarayanan, Pritzel, Blundell. *Simple and Scalable Predictive Uncertainty
  Estimation using Deep Ensembles.* NeurIPS 2017.
- [GEY2017] Geifman, El-Yaniv. *Selective Classification for Deep Neural Networks.* NeurIPS 2017.
  (Risk-coverage / AURC.)
- [BH1995] Benjamini, Hochberg. *Controlling the False Discovery Rate.* JRSS-B, 1995.
- [NAVSIM2024] Dauner et al. *NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and
  Benchmarking.* NeurIPS 2024. (PDM score.)
- [RI2026] Cross-benchmark open-loop vs closed-loop ranking-inversion study, arXiv:2605.00066,
  2026.
- [RFS2026] Waymo Open Dataset End-to-End driving and Rater Feedback Score, arXiv:2510.26125,
  2026.

*See `docs/RELATED_WORK.md` for verification grades on the cited claims and `docs/MATHEMATICS.md`
for the formal definition of every signal and statistic.*
