# Phase 2B — NAVSIM closed-loop validity bridge (pre-registered no-go)

The experiment behind [`docs/PHASE2B_FINDINGS.md`](../../docs/PHASE2B_FINDINGS.md): does a cheap
learned predictor of closed-loop PDMS recover the closed-loop ranking better than the open-loop
ADE metric? Tested where supervision is abundant (the PDM simulator), so the Phase-2A data wall
does not apply.

| File | Role |
|---|---|
| `score_bridge.py` | per-trajectory closed-loop PDMS scorer (NAVSIM PDM simulator; 6 ego-MLP members + heuristics) |
| `analyze.py` | the validity-bridge verdict: cheap learned predictor vs open-loop ADE at recovering closed-loop ranking, paired drive-cluster bootstrap |
| `bridge_data.json` | 760 scenes / 33 drives, 6,080 scored trajectories — **derived only** (our trajectories + simulator PDMS + precomputed ADE; no ground-truth trajectory, no frames) |
| `verdict.json` | computed statistics |

## Verdict

**No decisive win (no-go).** Learned (cheap geometry) vs ADE: overall Spearman 0.144 vs 0.404;
within-scene Kendall-τ 0.083 vs 0.060; paired difference +0.023, 95% CI [−0.040, +0.100] —
inconclusive, stable across 280→760 scenes. Both are weak because closed-loop PDMS is
**scene-interactive** and cheap trajectory geometry cannot see the scene. Reproduce:

```bash
python analyze.py bridge_data.json
```
