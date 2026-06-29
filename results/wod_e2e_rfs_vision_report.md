> **CORRECTION (2026-06-29, see [P2g](wod_e2e_rfs_surround_report.md)):** the headline
> claim below — that perception grounding *significantly* beats ego (paired Δρ = +0.122,
> 95% CI [+0.003, +0.242]) — **does not replicate and is retracted.** It was a single
> high-variance instantiation whose CI cleared zero by only +0.003. A 20-seed-set stability
> study (P2g) shows the rungs overlap and the point estimates favor ego (P(vision>ego) ≈
> 0.30). The robust result is **no improvement from frozen-encoder perception grounding**;
> the human-RFS correlation from label-free disagreement is ρ ≈ 0.18, below the 0.30 bar.
> This file is kept for the record, not as a finding.

# P2f — perception grounding (PRELIMINARY, superseded by P2g): a single instantiation suggested a lift

Date: 2026-06-29. Compute: GCP n2-standard-16, CPU. Data: WOD-E2E validation split,
same 479 rater-labeled frames as P2e. Scored by Waymo's official
`rater_feedback_utils.get_rater_feedback_score` and this repo's tested statistics.

## The question P2e left open

P2e found the cheap label-free signal predicts human RFS only weakly (ρ = 0.15, below the
pre-registered 0.30 bar). The diagnosis was that the planner is **ego-status only** — its
ensemble members disagree on ego kinematics, not on *scene difficulty*. P2f tests that
diagnosis directly: **does grounding the ensemble in perception strengthen the signal?**

## Method — a controlled A/B on identical frames

Both ensembles are trained on the **same 13,950 frames** and evaluated on the **same 479
rater frames**; the only difference is the input features:

- **ego** — `ego_status(12)` = ego velocity/accel + driving intent (reproduces the P2e null here).
- **vis** — `ego_status(12) + DINOv2(384)`, a frozen `vit_small_patch14_dinov2` embedding of
  the **front camera** (no driving fine-tuning, single camera — deliberately a lower bound).

For each rater frame and each arm: predict K = 4 trajectories, measure ensemble
disagreement (mean pairwise displacement), score the ensemble-mean trajectory's RFS, and
record ADE. Correlations target `neg_RFS = −RFS` with drive-level (per-shard)
cluster-bootstrap CIs (10k). The decisive comparison is **paired**: Δρ = ρ_vis − ρ_ego,
recomputed on the same resampled shards each bootstrap iteration.

## Result

479 rater frames / 93 drives. Minimum detectable |ρ| at this n = 0.128.

| Arm | features | Spearman ρ (disagreement vs neg-RFS) | 95% CI (cluster) | p |
|---|---|---|---|---|
| ego-only (P2e null) | ego_status(12) | +0.100 | [+0.017, +0.179] | 0.030 |
| **DINOv2-grounded** | ego + front-cam(384) | **+0.223** | [+0.126, +0.315] | 0.0001 |

**Paired test — does vision beat ego?**

| Δρ = ρ_vis − ρ_ego | 95% CI (paired cluster bootstrap) | P(vis > ego) | verdict |
|---|---|---|---|
| **+0.122** | **[+0.003, +0.242]** | 0.978 | **decisive — CI excludes 0** |

Triage of the worst-rated frames (bottom-quartile RFS): AUROC ego 0.560 → **vis 0.626**.

## Honest reading

1. **Perception grounding causally and significantly improves the label-free signal.** The
   paired bootstrap — which holds the frames, the statistic, and the target fixed and
   varies only the feature set — puts Δρ at +0.122 with a 95% CI that excludes zero
   (P = 0.978). The signal's prediction of *human* ratings **more than doubles** (0.10 →
   0.22) when the ensemble can see the scene. P2e's weakness was the planner's blindness,
   not the hypothesis.
2. **It is decisive but marginal — reported exactly.** The paired CI lower bound is +0.003.
   Significant at the 95% level, but barely; we state the number rather than rounding the
   nuance away.
3. **Still below the 0.30 bar — and this is a lower bound.** The vision arm (ρ = 0.223) has
   not cleared H1's threshold, but it used a **single front camera**, a **frozen**
   ImageNet/LVD-pretrained encoder with **no driving fine-tuning**, and a tiny MLP head. A
   full perception stack (all eight cameras + LiDAR, driving-trained) is the obvious next
   lever, and the trajectory — more perception, more signal — now has a measured slope, not
   just a hypothesis.

## Where this leaves the arc

| Target | Signal behavior |
|---|---|
| Open-loop ADE (P2a) | strong — ρ = 0.70 |
| Closed-loop PDMS score (P2b) | null — ρ ≈ 0 |
| Closed-loop gate events (P2c) | decisive — AUROC ~0.8 |
| Human RFS, ego-only (P2e) | weak — ρ = 0.10–0.15 |
| **Human RFS, perception-grounded (P2f)** | **ρ = 0.22, and paired-significantly > ego** |

The headline is not the absolute number — it is the **slope**: giving the ensemble eyes
moves the human-rating correlation up by a margin whose confidence interval excludes zero.

## Reproduce

```bash
# VM with the WOD-E2E val split + waymo toolkit + torch/timm (env `wod`):
python experiments/wod_e2e_rfs/run_rfs_vision.py        # parse 93 shards, embed front cam (DINOv2),
                                                        # train ego + vision ensembles, score RFS
# locally, with the repo's tested statistics:
python experiments/wod_e2e_rfs/analyze_vision.py experiments/wod_e2e_rfs/wod_rfs_vision_out.json
```

No Waymo frames redistributed — only per-frame derived scalars
(`ego_dis/rfs/ade`, `vis_dis/rfs/ade`, `init_speed`, `shard`).
