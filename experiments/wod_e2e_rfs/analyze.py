"""WOD-E2E RFS analysis — the clean verdict.

Reads the per-frame derived scalars (disagreement, RFS, ADE, init_speed, shard)
produced on the VM by experiments/wod_e2e_rfs/run_rfs.py and computes the
pre-registered statistics with the repo's tested scoring.py:

  H1  does a label-free signal predict per-frame RFS?  (Spearman rho >= 0.3, q < 0.05)
  H3  does it triage low-rated frames better than chance? (AUROC / AP / E-AURC)

All correlations target neg_rfs = -RFS (positive rho => signal flags worse human
ratings). CIs are drive-level (by shard) cluster bootstrap. No Waymo frames are
redistributed; this reads only derived scalars.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from perceptionproof.scoring import (  # noqa: E402
    benjamini_hochberg,
    failure_mining,
    min_detectable_rho,
    risk_coverage,
    spearman_with_ci,
)

data = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "wod_rfs_out.json"))
rows = data["rows"]
dis = np.array([r["disagreement"] for r in rows], float)
ade = np.array([r["ade"] for r in rows], float)
rfs = np.array([r["rfs"] for r in rows], float)
shard = np.array([r["shard"] for r in rows])
neg = -rfs
n = len(rows)

print(f"n_frames={n}  n_drives(shards)={len(set(shard.tolist()))}  n_train={data['n_train']}  K={data['K']}")
print(f"RFS: mean={rfs.mean():.3f} sd={rfs.std():.3f} min={rfs.min():.1f} max={rfs.max():.1f}")
print(f"min detectable |rho| at n={n}, power 0.8: {min_detectable_rho(n):.3f}\n")

print("== H1: Spearman vs neg-RFS (drive-cluster bootstrap, 10k) ==")
res = {}
for name, sig in [("disagreement (label-free)", dis), ("ADE-to-human (needs label)", ade)]:
    r = spearman_with_ci(sig, neg, shard)
    res[name] = r
    flag = "PASS" if (r["rho"] >= 0.3 and r["ci_low"] > 0) else ("weak+" if r["ci_low"] > 0 else "ns")
    print(f"  {name:28s} rho={r['rho']:+.3f} [{r['ci_low']:+.3f},{r['ci_high']:+.3f}] "
          f"p={r['p_value']:.4f}  -> H1(>=0.3)? {flag}")

print("\n== H3: triage low-rated frames (failure = bottom-quartile RFS) ==")
thr = np.quantile(rfs, 0.25)
fail = (rfs <= thr).astype(int)
print(f"  failure threshold RFS<={thr:.2f}  base_rate={fail.mean():.3f}")
for name, sig in [("disagreement", dis), ("ADE", ade)]:
    fm = failure_mining(sig, fail)
    rc = risk_coverage(sig, neg)
    print(f"  {name:12s} AUROC={fm['auroc']:.3f} AP={fm['ap']:.3f} (base {fm['base_rate']:.3f})  "
          f"E-AURC={rc['e_aurc']:.3f}")

print("\n== open-loop coupling sanity ==")
c = spearman_with_ci(dis, ade, shard)
print(f"  disagreement vs ADE: rho={c['rho']:+.3f} [{c['ci_low']:+.3f},{c['ci_high']:+.3f}] p={c['p_value']:.4f}")

print("\n== BH-FDR over H1 p-values (q=0.05) ==")
pv = {k: v["p_value"] for k, v in res.items()}
for k, ok in benjamini_hochberg(pv).items():
    print(f"  {k:28s} q<0.05 significant? {ok}")
