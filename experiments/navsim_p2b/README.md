# Experiment P2b — closed-loop PDMS on NAVSIM

The decisive, non-structural test: does ensemble disagreement predict the **closed-loop** PDM
score (an independent simulator-derived outcome), not just the structurally-coupled open-loop
ADE of P2a?

## Pipeline

1. **Metric caching** — build NAVSIM metric caches (PDM simulator state) for the scored scenes:
   ```bash
   python navsim/planning/script/run_metric_caching.py \
     train_test_split=navtrain 'train_test_split.scene_filter.tokens=[...]' \
     train_test_split.scene_filter.log_names=null
   ```
   (Data must be at `OPENSCENE_DATA_ROOT/navsim_logs/<split>/`; symlink the trainval logs there.)
2. **Score** — `score_pdms.py` trains a native-architecture ego-status MLP ensemble on scenes
   disjoint from the cached tokens, then for each cached token computes every member's trajectory
   and its PDMS via `navsim.evaluate.pdm_score`, dumping `{token, trajs, pdms}`.
3. **Analyze** — `analyze_pdms.py` computes S1 disagreement and correlates it against PDMS with
   the unit-tested `perceptionproof.scoring` (cluster-bootstrapped by log).

```bash
python analyze_pdms.py ~/pp_pdms.json
```

## Why this is the decisive test

PDMS comes from a simulator (collision, drivable-area, time-to-collision, progress, comfort), not
from the ensemble's own trajectories — so a strong **negative** correlation (high disagreement ↔
low PDMS) cannot be the algebraic variance↔mean-error artifact that the P2a leave-one-out test
already controlled for. It is the closed-loop-aligned evidence the thesis rests on.

## Reproduce from committed data (no dataset needed)

```bash
python analyze_pdms.py pp_pdms.json    # rho=-0.074 [-0.396,0.285] (the pre-registered null)
```

## Honesty / data

`pp_pdms.json` holds only **derived** data: segment ids, our ensemble's predicted trajectories,
and the simulator-computed PDM score per scene. No OpenScene/nuPlan media or labels are
redistributed. See `../../DATA_LICENSES.md`.
