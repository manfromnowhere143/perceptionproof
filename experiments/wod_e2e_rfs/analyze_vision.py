"""WOD-E2E RFS, perception-grounded ensemble (P2f) — the clean A/B verdict.

Compares the ego-only ensemble (the P2e null) against a DINOv2-grounded ensemble on
the SAME 479 rater frames: does scene-aware disagreement predict human RFS better?

- Per-arm: Spearman(disagreement, neg-RFS) with drive-cluster bootstrap CI.
- Paired test: Delta-rho = rho_vis - rho_ego with a paired cluster bootstrap (resample
  shards once per iteration, recompute BOTH rhos on the same resampled frames) — the
  honest way to ask whether vision grounding *significantly* helps.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from perceptionproof.scoring import failure_mining, min_detectable_rho, spearman_with_ci  # noqa: E402

data = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "wod_rfs_vision_out.json"))
rows = data["rows"]
shard = np.array([r["shard"] for r in rows])
neg = {a: -np.array([r[f"{a}_rfs"] for r in rows], float) for a in ("ego", "vis")}
dis = {a: np.array([r[f"{a}_dis"] for r in rows], float) for a in ("ego", "vis")}
n = len(rows)

print(f"n_frames={n}  drives={len(set(shard.tolist()))}  n_train={data['n_train']}  emb_dim={data['emb_dim']}")
print(f"min detectable |rho| at n={n}: {min_detectable_rho(n):.3f}\n")

print("== per-arm: disagreement vs neg-RFS (drive-cluster bootstrap 10k) ==")
arm = {}
for a in ("ego", "vis"):
    r = spearman_with_ci(dis[a], neg[a], shard)
    arm[a] = r
    bar = "PASS(>=0.3)" if (r["rho"] >= 0.3 and r["ci_low"] > 0) else ("real(<0.3)" if r["ci_low"] > 0 else "ns")
    label = "ego-only (P2e null)" if a == "ego" else "DINOv2-grounded"
    print(f"  {label:22s} rho={r['rho']:+.3f} [{r['ci_low']:+.3f},{r['ci_high']:+.3f}] p={r['p_value']:.4f}  {bar}")

print("\n== paired: does vision beat ego? Delta-rho = rho_vis - rho_ego ==")
rng = np.random.default_rng(20260629)
uniq = np.unique(shard)
idx_by = {d: np.where(shard == d)[0] for d in uniq}
d_obs = arm["vis"]["rho"] - arm["ego"]["rho"]
boots = []
for _ in range(10000):
    samp = rng.choice(uniq, size=len(uniq), replace=True)
    rowsel = np.concatenate([idx_by[d] for d in samp])
    if rowsel.size < 5:
        continue
    rv = spearmanr(dis["vis"][rowsel], neg["vis"][rowsel]).statistic
    re = spearmanr(dis["ego"][rowsel], neg["ego"][rowsel]).statistic
    if not (np.isnan(rv) or np.isnan(re)):
        boots.append(rv - re)
lo, hi = np.percentile(boots, [2.5, 97.5])
frac = float(np.mean(np.array(boots) > 0))
verdict = "DECISIVE: vision > ego (CI excludes 0)" if lo > 0 else "inconclusive (CI includes 0)"
print(f"  Delta-rho = {d_obs:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  P(vis>ego)={frac:.3f}  -> {verdict}")

print("\n== triage worst-rated frames (bottom-quartile RFS per arm) ==")
for a in ("ego", "vis"):
    rfs = -neg[a]
    fail = (rfs <= np.quantile(rfs, 0.25)).astype(int)
    fm = failure_mining(dis[a], fail)
    print(f"  {a:4s} AUROC={fm['auroc']:.3f} AP={fm['ap']:.3f} (base {fm['base_rate']:.3f})")
