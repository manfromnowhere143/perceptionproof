# P2g — surround view + a stability study that corrects P2f: perception grounding does NOT robustly help

Date: 2026-06-29. Compute: GCP n2-standard-16, CPU. Data: WOD-E2E validation, 479 rater
frames. This phase pushed the perception lever harder (all 8 cameras) and, in doing so,
discovered that the P2f improvement was not stable. It is reported as a **correction**.

## What happened

P2f added a front camera and a single ensemble showed disagreement-vs-RFS rising from
ρ = 0.10 (ego) to 0.22 (vision), with a paired bootstrap whose CI *barely* excluded zero
(lower bound +0.003). P2g set out to extend that slope with the **full 8-camera field of
view**, dimensionality-controlled via PCA-256.

The single-instantiation surround run did not extend the slope — it **inverted** it:

| rung (single instantiation) | ρ (disagreement vs neg-RFS) | 95% CI (cluster) |
|---|---|---|
| ego | +0.292 | [0.221, 0.357] |
| front | +0.159 | [0.068, 0.248] |
| surround | +0.104 | [0.021, 0.188] |

Here *ego* — the blind baseline — was the **best**, and it landed at 0.29, far from the
0.10 it showed in P2f. The same nominal model, two runs, ρ from 0.10 to 0.29. That swing
is the tell: at this effect size the disagreement-vs-RFS correlation is sensitive to which
seeds and training sample the ensemble drew. A single number cannot be trusted.

## The stability study (the decisive test)

Using the cached embeddings (no re-parse), each rung was re-trained over **20 independent
seed-sets**, with the *same* seeds across rungs at each instantiation so the comparison is
paired and isolates the feature set. The distribution of ρ:

| rung | mean ρ | sd | range |
|---|---|---|---|
| **ego** | **+0.176** | 0.048 | [0.075, 0.262] |
| front | +0.161 | 0.026 | [0.112, 0.218] |
| surround | +0.124 | 0.023 | [0.084, 0.168] |

Paired per-instantiation deltas:

| comparison | mean Δ | P(Δ>0) | 95% interval |
|---|---|---|---|
| front − ego | −0.016 | 0.30 | [−0.092, +0.076] |
| surround − ego | −0.053 | 0.10 | [−0.133, +0.033] |
| surround − front | −0.037 | 0.20 | [−0.109, +0.030] |

**Every paired interval straddles zero, and the point estimates favor ego.** Adding a
frozen image encoder — front *or* full surround — does **not** robustly improve the
label-free signal's prediction of human RFS. If anything it slightly hurts (more input
dimensions, no added RFS-aligned information, noisier ensemble disagreement).

## Correction to P2f

**The P2f claim that perception grounding "decisively" beats ego (paired Δρ CI excludes 0)
does not replicate and is retracted.** It was a single high-variance draw whose CI cleared
zero by +0.003 — exactly the margin this stability study shows is noise. The P2f report
now carries a correction banner; the result file is kept for the record, not as a finding.

## The robust statement

On the human-rated WOD-E2E benchmark, a label-free ensemble-disagreement signal predicts
the Rater Feedback Score **weakly but stably at ρ ≈ 0.18** (mean over 20 instantiations,
ego-status ensemble) — real (every instantiation > 0) but **below the pre-registered 0.30
bar**, so **H1 is not met**. Frozen-encoder perception grounding does not change that. The
oracle anchor (ADE, which needs the human label) reaches ρ ≈ 0.40, so the score *is*
predictable; cheap label-free disagreement from these weak planners simply does not recover
it.

## What this does and does not rule out

- **Ruled out (for this setup):** that *adding a frozen pretrained image embedding as MLP
  input features* meaningfully improves label-free RFS prediction. It does not, robustly.
- **Not ruled out:** that a *jointly-trained end-to-end vision/LiDAR planner* (whose
  ensemble disagreement reflects learned driving uncertainty, not frozen ImageNet/LVD
  features) could do better. That needs a GPU and real planner training — a separate,
  larger experiment, now clearly motivated and honestly scoped.

## Methodological note (why we caught it)

The lesson is procedural: **report distributions, not single draws, for small effects.** The
embedding cache (`wod_surround_emb.npz`) makes the stability study cheap and is kept so any
future ablation re-runs in minutes without touching the 243 GB of frames.

## Reproduce

```bash
python experiments/wod_e2e_rfs/run_rfs_surround.py     # parse + 8-cam embed -> cache + single-instantiation JSON
python experiments/wod_e2e_rfs/stability_study.py      # 20 seed-sets off the cache (set BLAS threads = 1!)
python experiments/wod_e2e_rfs/analyze_surround.py     # single-instantiation CIs + the stability verdict
```

Engineering note: the stability study must pin BLAS threads
(`OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1`) before NumPy imports — 12 workers × multithreaded
BLAS drove load to ~190 and wedged the box. No Waymo frames redistributed.
