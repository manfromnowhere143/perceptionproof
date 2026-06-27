"""Score a NAVSIM ensemble run with PerceptionProof's tested signals + scoring.

Usage: python analyze.py <pp_result.json>

Does ensemble trajectory disagreement (S1) predict open-loop error (ADE vs the human future)?
Uses the same code paths covered by the unit tests.
"""

import json
import sys

import numpy as np

from perceptionproof.scoring import failure_mining, risk_coverage, spearman_with_ci
from perceptionproof.signals import s1_ensemble_disagreement, trajectory_distance
from perceptionproof.types import ModelOutput, TrajectoryMode


def main(path: str) -> None:
    d = json.load(open(path))
    rows, K = d["rows"], d["K"]

    # median-heuristic sigma for the S1 MMD kernel
    pair_d = []
    for r in rows:
        T = [np.asarray(t, float) for t in r["trajs"]]
        for i in range(len(T)):
            for j in range(i + 1, len(T)):
                pair_d.append(trajectory_distance(T[i], T[j]))
    sigma = float(np.median(pair_d)) or 1.0
    print(f"[analyze] scenes={len(rows)} K={K} sigma(median)={sigma:.3f}")

    g, ade, fde, logs = [], [], [], []
    for r in rows:
        trajs = [np.asarray(t, float) for t in r["trajs"]]
        human = np.asarray(r["human"], float)
        outs = [ModelOutput(model_id=f"m{k}", weights_sha256="x",
                            trajectory_modes=[TrajectoryMode(waypoints=trajs[k], weight=1.0)]) for k in range(K)]
        g.append(s1_ensemble_disagreement(outs, sigma=sigma))
        errs = [np.linalg.norm(trajs[k] - human, axis=1) for k in range(K)]
        ade.append(float(np.mean(errs)))
        fde.append(float(np.mean([e[-1] for e in errs])))
        logs.append(r["log"])

    g, ade, fde, logs = np.array(g), np.array(ade), np.array(fde), np.array(logs)
    theta = float(np.median(ade))
    failure = (ade > theta).astype(int)

    print(f"[analyze] ADE mean={ade.mean():.2f}m median={theta:.2f}m  FDE mean={fde.mean():.2f}m  drives={len(set(logs))}")
    sp = spearman_with_ci(g, ade, logs, n_boot=2000, n_perm=2000, seed=20260627)
    print(f"[H1] Spearman rho={sp['rho']:.3f}  CI[{sp['ci_low']:.3f},{sp['ci_high']:.3f}]  p={sp['p_value']:.4f}  n={sp['n']}")
    fm = failure_mining(g, failure, ks=(50, 100))
    print(f"[H3] AUROC={fm['auroc']:.3f}  AP={fm['ap']:.3f}  base={fm['base_rate']:.3f}  prec@k={fm['precision_at_k']}")
    rc = risk_coverage(g, ade)
    print(f"[sel] AURC={rc['aurc']:.3f}  E-AURC={rc['e_aurc']:.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pp_result.json")
