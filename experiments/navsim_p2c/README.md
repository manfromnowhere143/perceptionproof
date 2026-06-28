# Experiment P2c — label-free signals vs PDMS gate events (powered)

The decisive, properly-powered test behind [`results/navsim_p2c_report.md`](../../results/navsim_p2c_report.md).
Designed from the verified literature (`docs/RELATED_WORK.md`): PDMS is gated by binary NC
(collision) / DAC (off-road) multipliers, so the target must be the **binary gate events**, not the
smooth score — and the signal should be **collision-aligned**, not displacement.

## Result (1,317 scenes, 55 drives)

- Label-free signals predict the gate events at **AUROC 0.77–0.83** (CIs exclude chance) — both
  collision and off-road — where the same disagreement signal was at chance against the aggregate
  PDMS (P2b). **Decisive: reframing the target to the gates unlocks closed-loop predictability.**
- Paired AUROC difference (collision-geometry vs disagreement) is **inconclusive** on both gates —
  no signal dominates. The reframing matters; the specific signal does not, at this power.

## Pipeline (runs on a CPU box, no sensors, no GPU)

```bash
# 1. drive-stratified tokens (PER_LOG x LOGS_WANTED), for tight cluster-bootstrap CIs
PER_LOG=24 LOGS_WANTED=55 python gen_tokens.py            # -> ~/p2c_tokens.txt

# 2. metric-cache those scenes (PDM simulator state)
python navsim/planning/script/run_metric_caching.py train_test_split=navtrain \
  "train_test_split.scene_filter.tokens=$(cat ~/p2c_tokens.txt)" train_test_split.scene_filter.log_names=null

# 3. parallel score: collision-geometry signal + PDM gate metrics, one PDM sim per core
K=4 EPOCHS=300 NPROC=8 OUT=~/pp_p2c.json python score_parallel.py

# 4. clean verdict: per-gate AUROC + paired AUROC-difference (bootstrap by drive)
python analyze.py ~/pp_p2c.json
```

## Signals (label-free, from the metric cache's own geometry)

- `collision_risk` — min distance ego→agents over the horizon (aligned with NC).
- `off_road` — fraction of ego **footprint** corners outside the drivable area (aligned with DAC).
- `disagreement` — ensemble trajectory spread (S1), the displacement baseline.

## Engineering notes

- Scoring is parallelized (one PDM simulation per core); the giant scene-index is freed before
  forking workers and workers are recycled (`maxtasksperchild`) to avoid OOM.
- Data layout: symlink the trainval logs into `OPENSCENE_DATA_ROOT/navsim_logs/<split>/`.
- No OpenScene/nuPlan data is redistributed (see `../../DATA_LICENSES.md`).
