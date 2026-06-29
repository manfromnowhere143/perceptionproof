# WOD-E2E Rater Feedback Score (P2e)

Tests whether a label-free ensemble-disagreement signal predicts the **Waymo Rater
Feedback Score** — the official human-rated long-tail driving benchmark. Result and
honest caveats: [`../../results/wod_e2e_rfs_report.md`](../../results/wod_e2e_rfs_report.md).

## Files

| File | Role |
|---|---|
| `validate_rfs_api.py` | Confirms the official `get_rater_feedback_score` call on one rater frame (sanity that the proto fields and trust-region scoring are wired correctly). |
| `run_rfs.py` | **P2e** — parses all 93 WOD-E2E val shards (parallel), trains a 4-seed ego-status MLP ensemble, and for each rater frame writes the derived scalars (`disagreement`, `rfs`, `ade`, `init_speed`, `shard`) to `wod_rfs_out.json`. |
| `analyze.py` | P2e statistics with the repo's tested `scoring.py` (Spearman + drive-cluster bootstrap, failure-mining AUROC/AP, E-AURC, BH-FDR). |
| `run_rfs_vision.py` | **P2f** — same parse, but also embeds the front camera with a frozen DINOv2 and trains BOTH an ego-only and an ego+vision ensemble (A/B on identical frames) → `wod_rfs_vision_out.json`. |
| `analyze_vision.py` | P2f verdict: per-arm Spearman + a **paired** cluster-bootstrap of Δρ = ρ_vis − ρ_ego (does perception grounding significantly help?). |
| `wod_rfs_out.json`, `wod_rfs_vision_out.json` | The derived scalars (479 rater frames). **No Waymo frames** — only our outputs. |

## Run

```bash
# VM with the WOD-E2E val split + waymo-open-dataset toolkit (env `wod`, py3.10):
python validate_rfs_api.py
python run_rfs.py                          # -> wod_rfs_out.json
# locally (repo .venv):
python analyze.py wod_rfs_out.json
```

## Data

WOD-E2E is a non-commercial research-licensed dataset (Waymo). This directory
redistributes none of it — only segment-level derived scalars. See
[`../../DATA_LICENSES.md`](../../DATA_LICENSES.md).

## Files (P2g)

| File | Role |
|---|---|
| `run_rfs_surround.py` | Parse + embed all 8 cameras (batched DINOv2), build the ego/front/surround ladder, cache embeddings to `wod_surround_emb.npz`, score RFS for one instantiation. |
| `stability_study.py` | Re-train each rung over 20 seed-sets off the cache (paired) — pin BLAS threads to 1. |
| `analyze_surround.py` | Single-instantiation cluster CIs + the stability distribution/verdict. |

## Headline

479 rater frames / 93 drives. **P2e:** ego-only disagreement predicts human RFS at
ρ ≈ 0.18 (mean over seed-sets) — real but **below the 0.3 bar** (H1 not met); oracle ADE
anchor ρ = 0.40, so RFS *is* predictable. **P2f → P2g (correction):** a single instantiation
suggested a front-camera DINOv2 lift (0.10→0.22), but a **20-seed-set stability study**
shows it was seed-noise — mean ρ ego 0.18 / front 0.16 / surround 0.12, every paired
interval straddles zero, point estimates favor ego (P(vision>ego) ≈ 0.10–0.30). **Frozen-
encoder perception grounding does not robustly help.** A jointly-trained end-to-end
vision/LiDAR planner (GPU) is not ruled out and is the honest next lever.
