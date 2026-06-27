# Data Licensing & Compliance

PerceptionProof uses third-party autonomous-driving datasets under their **research / non-commercial** licenses. This repository is engineered so that compliance is structural, not a promise.

## What this repo does and does not contain

- **Does NOT redistribute** any dataset frames, images, point clouds, or labels.
- **Does contain** only: (a) the **segment identifiers** we evaluate (`protocol/slices.json`), and (b) **our derived outputs** — signal values, scores, and signed receipts (`results/`).
- To reproduce, a user must independently obtain the datasets from their official sources under their own license acceptance.

## Datasets

| Dataset | Use here | License posture | Source |
|---|---|---|---|
| Waymo Open Dataset — E2E (WOD-E2E) | RFS-labelled long-tail segments | Waymo Dataset License — non-commercial research; no redistribution | waymo.com/open |
| NAVSIM / nuPlan | navtest open-loop ranking (PDMS) | nuPlan/NAVSIM research license — non-commercial | nuplan.org / NAVSIM repo |
| (optional) ZOD | long-range generalization checks | Zenseact Open Dataset license — research | zod.zenseact.com |

## Rules for contributors (binding)

1. Never commit dataset media or labels. CI/`.gitignore` blocks common dataset extensions and directories.
2. Commit only ids and derived artifacts.
3. If a dataset license forbids deriving/publishing a particular artifact, that artifact is not committed; document the exclusion here.
4. Commercial use of the **Aweb** backend does not extend any commercial right over these datasets; production runs that touch them remain bound by the dataset licenses.

## Code license

Repository code is Apache-2.0 (`LICENSE`). The data licenses above govern the data independently and are not superseded by Apache-2.0.
