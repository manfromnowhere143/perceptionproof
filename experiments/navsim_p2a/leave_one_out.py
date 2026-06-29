"""Independent-outcome strengthening for the NAVSIM run.

Usage: python leave_one_out.py <pp_result.json>

Signal = disagreement among ensemble members {0..K-2}; outcome = open-loop error of the
HELD-OUT member K-1. Signal and outcome come from disjoint models, so the algebraic
variance<->mean-error coupling (the main caveat of the coupled analysis) is removed. A
strong correlation here means disagreement tracks genuine scene difficulty, not an artifact.
Still open-loop; the closed-loop test is PDMS/RFS (P2b).
"""

import json
import sys

import numpy as np

from perceptionproof.scoring import failure_mining, spearman_with_ci
from perceptionproof.signals import s1_ensemble_disagreement, trajectory_distance
from perceptionproof.types import ModelOutput, TrajectoryMode


def main(path: str) -> None:
    d = json.load(open(path))
    rows, K = d["rows"], d["K"]
    sub = K - 1  # members used for disagreement; member K-1 is held out as the outcome

    pair_d = []
    for r in rows:
        T = [np.asarray(r["trajs"][k], float) for k in range(sub)]
        for i in range(len(T)):
            for j in range(i + 1, len(T)):
                pair_d.append(trajectory_distance(T[i], T[j]))
    sigma = float(np.median(pair_d)) or 1.0

    g, err, logs = [], [], []
    for r in rows:
        trajs = [np.asarray(t, float) for t in r["trajs"]]
        subset = [ModelOutput(model_id=f"m{k}", weights_sha256="x",
                              trajectory_modes=[TrajectoryMode(waypoints=trajs[k], weight=1.0)]) for k in range(sub)]
        g.append(s1_ensemble_disagreement(subset, sigma=sigma))
        # outcome = open-loop error of the held-out member (precomputed derived scalar)
        err.append(float(r["member_ade"][K - 1]))
        logs.append(r["log"])

    g, err, logs = np.array(g), np.array(err), np.array(logs)
    failure = (err > float(np.median(err))).astype(int)
    print(f"[loo] n={len(g)} disagreement_members={sub} held_out=1 sigma={sigma:.3f} drives={len(set(logs))}")
    sp = spearman_with_ci(g, err, logs, n_boot=2000, n_perm=2000, seed=20260627)
    print(f"[loo H1] rho={sp['rho']:.3f} CI[{sp['ci_low']:.3f},{sp['ci_high']:.3f}] p={sp['p_value']:.4f}")
    fm = failure_mining(g, failure, ks=(50, 100))
    print(f"[loo H3] AUROC={fm['auroc']:.3f} AP={fm['ap']:.3f} base={fm['base_rate']:.3f} prec@k={fm['precision_at_k']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pp_result.json")
