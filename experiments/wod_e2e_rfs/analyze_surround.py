"""P2g verdict — surround-view ladder + the stability study that corrects P2f.

(1) Single-instantiation cluster-bootstrap CIs for the ego/front/surround rungs
    (results/wod_rfs_surround_out.json).
(2) The stability distribution of rho over 20 independent seed-sets
    (stability_out.json) — the decisive, paired test of whether perception grounding
    *robustly* helps. This supersedes the single-draw P2f comparison.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from perceptionproof.scoring import spearman_with_ci  # noqa: E402

here = Path(__file__).resolve().parent
single = json.load(open(here / "wod_rfs_surround_out.json"))
stab = json.load(open(here / "stability_out.json"))
rows = single["rows"]
shard = np.array([r["shard"] for r in rows])

print("== (1) single instantiation — disagreement vs neg-RFS (cluster bootstrap) ==")
for tag in ("ego", "front", "surround"):
    sig = np.array([r[f"{tag}_dis"] for r in rows], float)
    neg = -np.array([r[f"{tag}_rfs"] for r in rows], float)
    r = spearman_with_ci(sig, neg, shard)
    print(f"  {tag:8s} rho={r['rho']:+.3f} [{r['ci_low']:+.3f},{r['ci_high']:+.3f}] p={r['p_value']:.4f}")
print("  (note: a SINGLE instantiation is one draw from the distribution below — do not over-read it)")

print("\n== (2) stability over 20 seed-sets (paired; same seeds across rungs per b) ==")
arr = {t: np.array(stab[t]) for t in ("ego", "front", "surround")}
for t in ("ego", "front", "surround"):
    a = arr[t]
    print(f"  {t:8s} mean rho={a.mean():+.3f}  sd={a.std():.3f}  range[{a.min():+.3f}, {a.max():+.3f}]")

print("\n== paired deltas (per-instantiation) ==")
ego, fr, su = arr["ego"], arr["front"], arr["surround"]
for name, dlt in [("front - ego", fr - ego), ("surround - ego", su - ego), ("surround - front", su - fr)]:
    print(f"  {name:16s} meanD={dlt.mean():+.3f}  P(>0)={float((dlt > 0).mean()):.2f}  "
          f"95% interval [{np.percentile(dlt, 2.5):+.3f}, {np.percentile(dlt, 97.5):+.3f}]")

print("\nVERDICT: perception grounding does NOT robustly improve the signal "
      "(P(vision>ego) <= 0.30). The P2f single-draw lift was within seed-noise. "
      "The robust human-RFS correlation from label-free disagreement is rho ~ 0.18, below the 0.30 bar.")
