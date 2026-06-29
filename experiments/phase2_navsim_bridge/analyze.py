"""Phase 2B — the validity-bridge verdict.

Reads pp_bridge.json (per-trajectory closed-loop PDMS for diverse trajectories per scene, plus
each scene's human future). Tests whether a CHEAP, deployable learned predictor of closed-loop
PDMS — using only trajectory geometry, no simulator — recovers the closed-loop ranking better
than the open-loop ADE-to-human metric the field uses (which is known to mis-rank closed-loop).

Primary metric: within-scene ranking of trajectories by closed-loop PDMS (Kendall-tau), the thing
an evaluation layer actually needs. Also overall Spearman. Held out by DRIVE. Paired
drive-cluster bootstrap on (learned - ADE) so the win is decisive or it isn't.
"""
# ruff: noqa: E702  (compact research script)
import json
import os
import sys

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

RNG = np.random.RandomState(17)


def feats(poses):
    p = np.asarray(poses, float)
    diffs = np.diff(p, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    head = np.arctan2((p[-1] - p[0])[1], (p[-1] - p[0])[0])
    perp = np.array([-np.sin(head), np.cos(head)])
    lat = (p - p[0]) @ perp
    ang = np.arctan2(diffs[:, 1], diffs[:, 0]) if len(diffs) else np.zeros(1)
    g = [float(seg.sum()), float(np.linalg.norm(p[-1] - p[0])), float(p[-1, 0]), float(p[-1, 1]),
         float(np.abs(lat).max()), float(lat.mean()), float(seg.mean()), float(seg.std()),
         float(np.abs(np.diff(ang)).sum()) if len(ang) > 1 else 0.0, float(seg.max()), float(seg.min())]
    return np.concatenate([p.reshape(-1), np.array(g, float)])


def main(path):
    d = json.load(open(path))
    rows = d["rows"]
    X, pdms, ade, sid, drive = [], [], [], [], []
    for si, r in enumerate(rows):
        for c in r["cands"]:
            if c.get("ade") is None:
                continue
            X.append(feats(c["poses"]))
            pdms.append(float(c["pdms"]))
            ade.append(float(c["ade"]))  # precomputed -ADE-to-human (derived scalar; higher=better)
            sid.append(si)
            drive.append(r["log"])
    X = np.array(X); pdms = np.array(pdms); ade = np.array(ade)
    sid = np.array(sid); drive = np.array(drive)
    print(f"scenes={len(rows)} candidates={len(X)} drives={len(set(drive))} "
          f"PDMS mean={pdms.mean():.3f} sd={pdms.std():.3f}")

    # held out by DRIVE
    udr = np.unique(drive); RNG.shuffle(udr)
    test_dr = set(udr[:max(1, int(0.3 * len(udr)))].tolist())
    te = np.array([dr in test_dr for dr in drive]); tr = ~te
    if te.sum() < 5 or tr.sum() < 50:
        print(f"insufficient split (train={tr.sum()} test={te.sum()}, drives={len(udr)}) — need more scenes")
        return
    model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=6, random_state=17)
    model.fit(X[tr], pdms[tr])
    pred = model.predict(X[te])
    yte, adte, ste, dte = pdms[te], ade[te], sid[te], drive[te]

    r_learn = spearmanr(pred, yte).correlation
    r_ade = spearmanr(adte, yte).correlation
    print(f"\n[overall held-out Spearman vs closed-loop PDMS]  learned={r_learn:+.3f}  ADE(open-loop)={r_ade:+.3f}")

    # within-scene ranking (Kendall-tau over each held-out scene's trajectories)
    def within_tau(score):
        ts = []
        for s in set(ste.tolist()):
            m = ste == s
            if m.sum() >= 3 and len(set(np.round(yte[m], 4))) > 1:
                t = kendalltau(score[m], yte[m]).correlation
                if not np.isnan(t):
                    ts.append(t)
        return np.array(ts)
    t_learn = within_tau(pred); t_ade = within_tau(adte)
    print(f"[within-scene Kendall-tau vs PDMS]  learned={t_learn.mean():+.3f}  ADE={t_ade.mean():+.3f}  "
          f"({len(t_learn)} scenes)")

    # paired drive-cluster bootstrap on the within-scene-tau difference (learned - ADE)
    # pair per scene; cluster by drive
    scene_drive = {s: dte[ste == s][0] for s in set(ste.tolist())}
    scenes = [s for s in set(ste.tolist())
              if (ste == s).sum() >= 3 and len(set(np.round(yte[ste == s], 4))) > 1]
    tl = {s: kendalltau(pred[ste == s], yte[ste == s]).correlation for s in scenes}
    ta = {s: kendalltau(adte[ste == s], yte[ste == s]).correlation for s in scenes}
    by_drive = {}
    for s in scenes:
        by_drive.setdefault(scene_drive[s], []).append(s)
    dr_keys = list(by_drive)
    diffs = []
    for _ in range(5000):
        samp = RNG.choice(len(dr_keys), len(dr_keys), replace=True)
        ss = [s for k in samp for s in by_drive[dr_keys[k]]]
        dl = np.nanmean([tl[s] for s in ss]); da = np.nanmean([ta[s] for s in ss])
        diffs.append(dl - da)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    dmean = float(np.nanmean([tl[s] for s in scenes]) - np.nanmean([ta[s] for s in scenes]))
    v = "DECISIVE: learned beats open-loop ADE (CI excl 0)" if lo > 0 else (
        "open-loop wins" if hi < 0 else "inconclusive")
    print(f"\n[CLEAN VERDICT] within-scene Kendall-tau, learned - ADE = {dmean:+.3f}  "
          f"95% CI[{lo:+.3f},{hi:+.3f}] -> {v}")
    json.dump({"r_learn": float(r_learn), "r_ade": float(r_ade),
               "tau_learn": float(t_learn.mean()), "tau_ade": float(t_ade.mean()),
               "delta": dmean, "delta_ci": [float(lo), float(hi)],
               "n_scenes": len(scenes), "n_cands": int(len(X))},
              open(os.path.expanduser("~/bridge_verdict.json") if False else
                   "bridge_verdict.json", "w"))
    print("WROTE bridge_verdict.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pp_bridge.json")
