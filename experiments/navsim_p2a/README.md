# Experiment P2a — NAVSIM ensemble disagreement vs open-loop error

Full, reproducible pipeline behind [`results/navsim_p2a_report.md`](../../results/navsim_p2a_report.md).
Everything here ran on a plain CPU VM (no GPU, no sensor data).

## Result (summary)

On 788 held-out real NAVSIM scenes (disjoint by-log split, 4-member ego-status MLP ensemble),
label-free disagreement predicts open-loop trajectory error: Spearman ρ = 0.699 [0.599, 0.750],
p = 0.0005; failure-mining AUROC = 0.855. See the report for the full table and the honest
caveats (open-loop not closed-loop; weak ego-status models; structural coupling; own split, not
the official navtest benchmark).

## Reproduce

Prereqs: a Linux box, ~30 GB disk, the NAVSIM/nuPlan research-license terms accepted.

```bash
# 1. environment (miniconda + NAVSIM devkit + nuplan-devkit)
bash setup_vm.sh

# 2. minimal data: maps + trainval logs only (no sensor blobs)
bash download_data.sh

# 3. train the ensemble and dump per-scene trajectories on a held-out (by-log) split
source ~/miniconda3/etc/profile.d/conda.sh && conda activate navsim
source ~/navsim_env.sh
cd ~/navsim_workspace/navsim
CAP=4000 K=4 EPOCHS=400 OUT=~/pp_result.json python /path/to/train_ensemble.py

# 4. score with PerceptionProof's tested signals + statistics
python analyze.py ~/pp_result.json
```

## Files

| File | Role |
|---|---|
| `setup_vm.sh` | miniconda + clone NAVSIM + conda env (accepts conda channel ToS) |
| `download_data.sh` | maps + trainval metadata logs only (sensor blobs skipped) |
| `train_ensemble.py` | K ego-status MLPs, by-log split, dumps trajectories + human future to JSON |
| `analyze.py` | computes S1 disagreement + ADE + correlation via `perceptionproof.signals`/`scoring` |

## Honesty / data

No OpenScene/nuPlan media or labels are committed — only code. The raw per-scene trajectory dump
is OpenScene-derived and is not redistributed (see `../../DATA_LICENSES.md`); only the aggregate
numbers in the report are published.
