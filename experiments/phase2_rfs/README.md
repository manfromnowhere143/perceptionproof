# Phase 2A — supervised RFS prediction (pre-registered null)

The experiments behind [`docs/PHASE2A_FINDINGS.md`](../../docs/PHASE2A_FINDINGS.md): a rigorous
test of whether a supervised model can predict / rank / improve the human Rater Feedback Score,
and the honest verdict that it is **data-walled on public data**.

| File | Role | Headline |
|---|---|---|
| `probe_g1.py` | predict RFS from synthetic-perturbation candidates | 0.736 — **inflated** (artifact) |
| `derisk_g1.py` | re-test on REAL planner trajectories | 0.573 vs ADE oracle 0.603; cross-dist 0.262 |
| `derisk_g1_scene.py` | add frozen DINOv2 scene perception | no help (Δ −0.034, CI [−0.100, +0.036]) |
| `g3_rerank.py` | reward-model reranking lift | none (reranked − default = −0.077, CI [−0.266, +0.112]) |

`*_out.json` are the committed derived scalars.

## The finding

A reference-free predictor *matches* the displacement (ADE) oracle (0.57 vs 0.60) but does not
beat it, frozen perception does not add, and a reward-model lift over a strong ensemble baseline is
blocked. **Mechanism:** the only abundant supervision (the recorded human future, on all 11,160
frames) *is* the displacement signal; the multimodal, trust-region structure that distinguishes
human preference (RFS) exists only on the 479 rated frames (the test set is hidden). So supervised
methods cannot learn human-preference-beyond-displacement at scale on public data — a characterized
data wall, and likely why no supervised RFS predictor exists.

## Run

CPU only, against the WOD-E2E-derived cache (rater references + ego features; no frames
redistributed). See the table above; each script writes its `*_out.json`.
