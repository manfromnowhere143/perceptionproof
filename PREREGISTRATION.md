# Pre-Registration

This file fixes the analysis **before** any result is computed. It is append-only: once a value is set and a study run has read it, it is not silently changed. Amendments are added as dated entries below, never overwrites. This is the discipline that makes a null result credible.

Frozen: 2026-06-27 (P1). Values marked `TBD@P2` are set during data/model wiring, before any correlation is computed, and then frozen.

## Hypotheses (binding decision rules — see docs/MATHEMATICS.md §6)

- **H1** — at least one signal $g_j$ has Spearman $\rho(g_j, -\mathrm{RFS}) \ge 0.30$, Benjamini–Hochberg $q < 0.05$.
- **H2** — signal-adjusted method ranking has strictly lower Kendall distance to ground truth than the open-loop metric, cluster-bootstrap CI of the difference excluding 0.
- **H3** — failure-mining Average Precision exceeds base rate AND E-AURC below random, $q < 0.05$.

## Frozen parameters

| Parameter | Symbol | Value |
|---|---|---|
| Failure threshold on RFS | $\theta_{\mathrm{RFS}}$ | `TBD@P2` (set from RFS distribution, e.g. lower-tertile cut, fixed before scoring) |
| Occupancy risk threshold | $\theta_{\mathrm{occ}}$ | `TBD@P2` |
| S3 entropy/conflict mix | $\alpha$ | 0.5 |
| Horizon discount | $\gamma$ | 1.0 (no discount in v0) |
| MMD kernel bandwidth | $\sigma$ | median-heuristic on the slice, computed once, then frozen |
| VLA rollouts | $K$ | 8 |
| Ranking penalty | $\lambda$ | `TBD@P2` (chosen by units, not by result) |
| Bootstrap replicates | $B$ | 10000 |
| Permutation replicates | $P$ | 10000 |
| FDR level | $q$ | 0.05 |
| RNG seed | — | 20260627 |

## Slice and power

- Dataset: WOD-E2E long-tail segments (RFS-labelled) + NAVSIM navtest for the inversion test.
- Slice size $n$: `TBD@P2`, chosen so the minimum detectable Spearman $\rho$ at $\alpha=0.05$, power 0.8 (Fisher-$z$, MATHEMATICS §5) is recorded here before results.
- Minimum detectable $\rho$ at chosen $n$: `TBD@P2`.
- Resampling unit: **drive** (cluster bootstrap); CV: drive-grouped K-fold.

## Models (frozen in protocol/models.lock.json at P2)

$M \ge 3$ public models spanning paradigms: one E2E planner, one occupancy/BEV model, one VLA reasoner. Exact ids + weight SHA-256 pinned in `protocol/models.lock.json` before any run.

## Amendments

- _none yet_
