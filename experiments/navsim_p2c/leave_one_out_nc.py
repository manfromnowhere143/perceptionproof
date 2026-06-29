"""P2d.1 — leave-one-out NC: does disagreement among the OTHER ensemble members predict the
deployed planner's collision gate? Removes the coupling caveat of P2c (the deployed planner,
whose NC is the outcome, is also inside the disagreement set).

Usage: python leave_one_out_nc.py <pp_p2c.json>

signal = S1 disagreement over members {1..K-1}; outcome = NC gate of the held-out member 0.
Signal and outcome come from disjoint models, so a strong AUROC is genuine scene difficulty,
not an algebraic artifact. Compares against the coupled (all-K) disagreement.
"""

import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from perceptionproof.signals import s1_ensemble_disagreement, trajectory_distance
from perceptionproof.types import ModelOutput, TrajectoryMode


def _disagreement(trajs, members, sigma):
    outs = [ModelOutput(model_id=f"m{k}", weights_sha256="x",
                        trajectory_modes=[TrajectoryMode(waypoints=trajs[k], weight=1.0)]) for k in members]
    return s1_ensemble_disagreement(outs, sigma=sigma)


def auroc_ci(signal, target, logs, n_boot=5000, seed=20260628):
    rng = np.random.default_rng(seed)
    uniq = np.unique(logs)
    idx = {u: np.where(logs == u)[0] for u in uniq}
    pt = roc_auc_score(target, signal)
    b = []
    for _ in range(n_boot):
        rb = np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq), replace=True)])
        if target[rb].sum() in (0, len(rb)):
            continue
        b.append(roc_auc_score(target[rb], signal[rb]))
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(pt), float(lo), float(hi)


def main(path):
    d = json.load(open(path))
    rows, K = d["rows"], d["K"]
    others = list(range(1, K))  # members 1..K-1 (exclude deployed member 0)

    pair = []
    for r in rows:
        T = [np.asarray(r["trajs"][k], float) for k in others]
        for i in range(len(T)):
            for j in range(i + 1, len(T)):
                pair.append(trajectory_distance(T[i], T[j]))
    sigma = float(np.median(pair)) or 1.0

    loo, coupled, nc, logs = [], [], [], []
    for r in rows:
        trajs = [np.asarray(t, float) for t in r["trajs"]]
        loo.append(_disagreement(trajs, others, sigma))      # disjoint from member 0
        coupled.append(_disagreement(trajs, range(K), sigma))  # all K (P2c)
        nc.append(int(r["nc"] < 1))
        logs.append(r["log"])
    loo, coupled, nc, logs = np.array(loo), np.array(coupled), np.array(nc), np.array(logs)

    print(f"n={len(rows)} drives={len(set(logs))} NC-events={nc.sum()}")
    a, lo, hi = auroc_ci(loo, nc, logs)
    print(f"  leave-one-out disagreement{{1..{K - 1}}} -> NC(member 0)  AUROC={a:.3f} CI[{lo:.3f},{hi:.3f}]")
    a2, lo2, hi2 = auroc_ci(coupled, nc, logs)
    print(f"  coupled disagreement{{0..{K - 1}}} -> NC (P2c)            AUROC={a2:.3f} CI[{lo2:.3f},{hi2:.3f}]")
    print("  verdict:", "holds (disjoint AUROC excludes chance) -> not a coupling artifact"
          if lo > 0.5 else "weakened under leave-one-out")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pp_p2c_scaled.json")
