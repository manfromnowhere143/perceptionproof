"""P2c analysis: do collision-aligned label-free signals predict the PDMS gate events, and do
they beat the displacement-disagreement baseline?

Usage: python analyze.py <pp_p2c.json>

Reports per-signal AUROC (drive-clustered bootstrap CI) for the NC (collision) and DAC (off-road)
gates, plus the CLEAN VERDICT: the paired AUROC difference (matched signal - baseline) with a
bootstrap-by-drive CI and two-sided p. Uses the unit-tested S1 disagreement for the baseline.
"""

import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from perceptionproof.signals import s1_ensemble_disagreement, trajectory_distance
from perceptionproof.types import ModelOutput, TrajectoryMode


def load(path):
    d = json.load(open(path))
    rows, K = d["rows"], d["K"]
    pair = []
    for r in rows:
        T = [np.asarray(t, float) for t in r["trajs"]]
        for i in range(len(T)):
            for j in range(i + 1, len(T)):
                pair.append(trajectory_distance(T[i], T[j]))
    sigma = float(np.median(pair)) or 1.0
    cols = {k: [] for k in ["disag", "crisk", "offroad", "nc", "dac", "pdms", "log"]}
    for r in rows:
        trajs = [np.asarray(t, float) for t in r["trajs"]]
        outs = [ModelOutput(model_id=f"m{k}", weights_sha256="x",
                            trajectory_modes=[TrajectoryMode(waypoints=trajs[k], weight=1.0)]) for k in range(K)]
        cols["disag"].append(s1_ensemble_disagreement(outs, sigma=sigma))
        for k in ("collision_risk", "off_road", "nc", "dac", "pdms"):
            cols[{"collision_risk": "crisk", "off_road": "offroad"}.get(k, k)].append(r[k])
        cols["log"].append(r["log"])
    return {k: np.array(v) for k, v in cols.items()}


def auroc_ci(signal, target, logs, n_boot=5000, seed=20260628):
    if target.sum() in (0, len(target)):
        return (float("nan"),) * 3
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


def paired(sig_a, sig_b, target, logs, n_boot=5000, seed=20260628):
    rng = np.random.default_rng(seed)
    uniq = np.unique(logs)
    idx = {u: np.where(logs == u)[0] for u in uniq}
    pt = roc_auc_score(target, sig_a) - roc_auc_score(target, sig_b)
    d = []
    for _ in range(n_boot):
        rb = np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq), replace=True)])
        if target[rb].sum() in (0, len(rb)):
            continue
        d.append(roc_auc_score(target[rb], sig_a[rb]) - roc_auc_score(target[rb], sig_b[rb]))
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(pt), float(lo), float(hi), float(p)


def main(path):
    c = load(path)
    logs = c["log"]
    nc, dac = (c["nc"] < 1).astype(int), (c["dac"] < 1).astype(int)
    print(f"n={len(logs)} drives={len(set(logs))}  NC={nc.sum()} DAC={dac.sum()}")
    for nm, sig, tgt in [("collision_risk->NC", c["crisk"], nc), ("disagreement->NC", c["disag"], nc),
                         ("off_road->DAC", c["offroad"], dac), ("disagreement->DAC", c["disag"], dac)]:
        a, lo, hi = auroc_ci(sig, tgt, logs)
        print(f"  {nm:22s} AUROC={a:.3f} CI[{lo:.3f},{hi:.3f}]")
    anyg = (c["pdms"] <= 1e-9).astype(int)  # PDMS=0 <=> any multiplicative gate failed
    print(f"\nany-gate (PDMS=0): events={int(anyg.sum())}")
    for nm, sig in [("disagreement->any", c["disag"]), ("collision_risk->any", c["crisk"]),
                    ("off_road->any", c["offroad"])]:
        a, lo, hi = auroc_ci(sig, anyg, logs)
        print(f"  {nm:22s} AUROC={a:.3f} CI[{lo:.3f},{hi:.3f}]")
    print("\nCLEAN VERDICT (paired AUROC difference, bootstrap by drive):")
    for nm, a, b, tgt in [("collision_risk - disagreement -> NC", c["crisk"], c["disag"], nc),
                          ("off_road - disagreement -> DAC", c["offroad"], c["disag"], dac)]:
        pt, lo, hi, p = paired(a, b, tgt, logs)
        v = "DECISIVE" if (lo > 0 or hi < 0) else "inconclusive"
        print(f"  {nm:36s} Δ={pt:+.3f} CI[{lo:+.3f},{hi:+.3f}] p={p:.3f} -> {v}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pp_p2c_scaled.json")
