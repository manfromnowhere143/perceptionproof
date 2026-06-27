"""Score the closed-loop PDMS run with PerceptionProof's tested signals + statistics.

Usage: python analyze_pdms.py <pp_pdms.json>

Does ensemble disagreement (S1) predict the closed-loop PDM score? PDMS is an independent,
simulator-derived outcome (not the structurally-coupled ADE), so a strong NEGATIVE correlation
here — high disagreement ↔ low PDMS — is the decisive, non-structural test of the signal.
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

    pair_d = []
    for r in rows:
        T = [np.asarray(t, float) for t in r["trajs"]]
        for i in range(len(T)):
            for j in range(i + 1, len(T)):
                pair_d.append(trajectory_distance(T[i], T[j]))
    sigma = float(np.median(pair_d)) or 1.0

    g, pdms, logs = [], [], []
    for r in rows:
        trajs = [np.asarray(t, float) for t in r["trajs"]]
        outs = [ModelOutput(model_id=f"m{k}", weights_sha256="x",
                            trajectory_modes=[TrajectoryMode(waypoints=trajs[k], weight=1.0)]) for k in range(K)]
        g.append(s1_ensemble_disagreement(outs, sigma=sigma))
        pdms.append(float(np.mean(r["pdms"])))   # scene PDMS = mean over ensemble members
        logs.append(r["log"])

    g, pdms, logs = np.array(g), np.array(pdms), np.array(logs)
    theta = float(np.median(pdms))
    failure = (pdms < theta).astype(int)          # low PDMS = failure
    print(f"[pdms] n={len(g)} sigma={sigma:.3f} drives={len(set(logs))}")
    print(f"[pdms] PDMS mean={pdms.mean():.3f} median={theta:.3f} min={pdms.min():.3f} max={pdms.max():.3f}")
    sp = spearman_with_ci(g, pdms, logs, n_boot=2000, n_perm=2000, seed=20260627)
    print(f"[H1] Spearman(disagreement, PDMS) rho={sp['rho']:.3f} CI[{sp['ci_low']:.3f},{sp['ci_high']:.3f}] p={sp['p_value']:.4f}  (expect NEGATIVE)")
    fm = failure_mining(g, failure, ks=(25, 50))
    print(f"[H3] AUROC={fm['auroc']:.3f} AP={fm['ap']:.3f} base={fm['base_rate']:.3f} prec@k={fm['precision_at_k']}")
    rc = risk_coverage(g, pdms.max() - pdms)
    print(f"[sel] AURC={rc['aurc']:.3f} E-AURC={rc['e_aurc']:.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pp_pdms.json")
