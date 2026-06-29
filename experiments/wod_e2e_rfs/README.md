# WOD-E2E Rater Feedback Score (P2e)

Tests whether a label-free ensemble-disagreement signal predicts the **Waymo Rater
Feedback Score** — the official human-rated long-tail driving benchmark. Result and
honest caveats: [`../../results/wod_e2e_rfs_report.md`](../../results/wod_e2e_rfs_report.md).

## Files

| File | Role |
|---|---|
| `validate_rfs_api.py` | Confirms the official `get_rater_feedback_score` call on one rater frame (sanity that the proto fields and trust-region scoring are wired correctly). |
| `run_rfs.py` | Parses all 93 WOD-E2E val shards (parallel), trains a 4-seed ego-status MLP ensemble, and for each rater frame writes the derived scalars (`disagreement`, `rfs`, `ade`, `init_speed`, `shard`) to `wod_rfs_out.json`. |
| `analyze.py` | Computes the pre-registered statistics with the repo's tested `scoring.py` (Spearman + drive-cluster bootstrap, failure-mining AUROC/AP, E-AURC, BH-FDR). |
| `wod_rfs_out.json` | The derived scalars (479 rater frames). **No Waymo frames** — only our outputs. |

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

## Headline

479 rater frames / 93 drives: label-free disagreement predicts worse human RFS at
ρ = 0.151 [0.063, 0.237] (real, BH q<0.05) but **below the pre-registered ρ ≥ 0.3 bar**
— H1 not met. The oracle anchor (ADE, needs the human label) reaches ρ = 0.40,
confirming RFS is genuinely predictable; an ego-status-only ensemble simply carries too
little scene information. A perception-grounded ensemble is the next power lever.
