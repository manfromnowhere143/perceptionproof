"""P2h verdict — does a JOINTLY-TRAINED vision ensemble's disagreement predict human RFS?

Reads p2h_out.json (K=6 end-to-end fine-tuned DINOv2 planners, scored by the official RFS).
Reports the full-K disagreement-vs-RFS with a drive-cluster bootstrap CI, the ADE oracle
anchor, and the member-bootstrap stability distribution — then sets it against the frozen
baselines (ego ρ≈0.18, frozen front-cam ρ≈0.16) and the 0.30 pre-registered bar.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from perceptionproof.scoring import min_detectable_rho, spearman_with_ci  # noqa: E402

here = Path(__file__).resolve().parent
d = json.load(open(here / "p2h_out.json"))
rows = d["full"]
shard = np.array([r["shard"] for r in rows])
dis = np.array([r["dis"] for r in rows], float)
rfs = np.array([r["rfs"] for r in rows], float)
ade = np.array([r["ade"] for r in rows], float)
neg = -rfs
sub = np.array(d["subset_rhos"], float)
n = len(rows)

print(f"jointly-trained vision ensemble  K={d['K']}  n_frames={n}  drives={len(set(shard.tolist()))}")
print(f"min detectable |rho| at n={n}: {min_detectable_rho(n):.3f}  RFS mean={rfs.mean():.2f} sd={rfs.std():.2f}\n")

rv = spearman_with_ci(dis, neg, shard)
ra = spearman_with_ci(ade, neg, shard)
print("== full-K (6 members) — drive-cluster bootstrap ==")
print(f"  disagreement vs neg-RFS : rho={rv['rho']:+.3f} [{rv['ci_low']:+.3f},{rv['ci_high']:+.3f}] p={rv['p_value']:.4f}")
print(f"  ADE(oracle) vs neg-RFS  : rho={ra['rho']:+.3f} [{ra['ci_low']:+.3f},{ra['ci_high']:+.3f}] p={ra['p_value']:.4f}")

print("\n== member-bootstrap stability (15 four-of-six sub-ensembles) ==")
print(f"  mean rho={sub.mean():+.3f}  sd={sub.std():.3f}  range[{sub.min():+.3f},{sub.max():+.3f}]")

print("\n== context: the same signal across representations (human RFS) ==")
print("  ego-status (frozen)        rho ~ 0.18   [P2g stability, sd 0.05]")
print("  frozen DINOv2 front-cam    rho ~ 0.16   [P2g stability]")
print("  frozen DINOv2 surround     rho ~ 0.12   [P2g stability]")
print(f"  JOINTLY-TRAINED vision     rho ~ {sub.mean():.2f}   [this study, sd {sub.std():.2f}]")
print("\nVERDICT: joint end-to-end training gives a slightly higher AND tighter signal "
      f"(rho {sub.mean():.2f}, sd {sub.std():.2f} vs ego sd ~0.05) and a genuinely better planner "
      f"(ADE-vs-RFS {ra['rho']:.2f}), but the disagreement-vs-RFS correlation still does NOT clear "
      "the 0.30 bar and overlaps the ego baseline. H1 remains not met even with a driving-trained "
      "vision ensemble.")
