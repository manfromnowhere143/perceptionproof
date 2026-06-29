"""Phase 2 / Phase A — G3: does a reference-free reward model lift a planner's actual RFS?

Train a reference-free quality model (geometry+ego -> RFS) on TRAIN scenes. On HELD-OUT
scenes, give a base planner (ego-MLP ensemble) a set of PLAUSIBLE proposals — its 6 ensemble
members + constant-velocity + small lateral/speed variants of the ensemble mean (NO broken
heuristics, NO reference leakage) — let the reward model pick the highest-rated proposal, and
measure the selected trajectory's TRUE RFS vs the planner's default (ensemble mean), random
selection, and the in-pool oracle/floor. Lift = reranker - default, bootstrap CI by scene.
"""
# ruff: noqa: E702  (compact exploratory probe scripts)
import json
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from waymo_open_dataset.metrics.python import rater_feedback_utils as rfu

CACHE = os.environ.get("CACHE", "/home/dev_alfred_ai_app/wod/wod_surround_emb.npz")
T, DT = 20, 0.25
RNG = np.random.RandomState(17)


def rfs_of(c, refs, labels, spd):
    out = rfu.get_rater_feedback_score(c.reshape(1, 1, T, 2), np.array([[1.0]]), [refs], [labels],
                                       np.array([spd]), frequency=4, length_seconds=5)
    return float(np.array(out["rater_feedback_score"]).ravel()[0])


def feats(c, ego):
    diffs = np.diff(c, axis=0); seg = np.linalg.norm(diffs, axis=1)
    head = np.arctan2((c[-1] - c[0])[1], (c[-1] - c[0])[0])
    perp = np.array([-np.sin(head), np.cos(head)]); lat = (c - c[0]) @ perp
    ang = np.arctan2(diffs[:, 1], diffs[:, 0])
    g = [float(seg.sum()), float(np.linalg.norm(c[-1] - c[0])), float(c[-1, 0]), float(c[-1, 1]),
         float(np.abs(lat).max()), float(lat.mean()), float(seg.mean()), float(seg.std()),
         float(np.abs(np.diff(ang)).sum()), float(seg.max()), float(seg.min())]
    return np.concatenate([c.reshape(-1), np.array(g, float), np.array(ego, float)])


def main():
    d = np.load(CACHE)
    ego_tr, Y_tr = d["ego_tr"], d["Y"]
    ego_r, spd_r = d["ego_r"], d["init_speed_r"]
    meta = json.load(open(CACHE + ".rater.json"))
    n = len(meta)
    gmean = Y_tr.mean(0).reshape(T, 2)
    sc = StandardScaler().fit(ego_tr); Xs = sc.transform(ego_tr)
    members = [MLPRegressor(hidden_layer_sizes=(128, 128), alpha=1e-4, max_iter=300, random_state=100 + k).fit(Xs, Y_tr)
               for k in range(6)]
    mp = np.stack([m.predict(sc.transform(ego_r)).reshape(n, T, 2) for m in members], 0)
    print(f"train={len(ego_tr)} rater={n}; ensemble ready", flush=True)

    uf = np.arange(n); RNG.shuffle(uf)
    test_f = set(uf[:int(0.3 * n)].tolist())

    # --- train the reference-free reward model on TRAIN scenes (broad pool incl. bad, to span RFS) ---
    Xtr, Ytr = [], []
    for i in range(n):
        if i in test_f:
            continue
        refs = [np.asarray(t, float) for t in meta[i]["rtrajs"]]; labels = np.asarray(meta[i]["rscores"], float)
        spd = float(spd_r[i]); ego = ego_r[i]; vx, vy = float(ego[0]), float(ego[1])
        steps = (np.arange(1, T + 1) * DT)[:, None]
        pool = [mp[k, i] for k in range(6)] + [steps * np.array([vx, vy]) * s for s in (0.5, 1.0, 1.5)]
        pool += [np.zeros((T, 2)) + 1e-3, np.mean(refs, 0), gmean.copy()]
        for c in pool:
            c = np.asarray(c, float)[:T]
            if c.shape[0] == T:
                Xtr.append(feats(c, ego)); Ytr.append(rfs_of(c, refs, labels, spd))
    reward = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=6, random_state=17)
    reward.fit(np.array(Xtr), np.array(Ytr))
    print(f"reward model trained on {len(Xtr)} train-scene samples", flush=True)

    # --- G3 eval on HELD-OUT scenes: PLAUSIBLE proposals only, no broken heuristics, no reference ---
    rows = []
    for i in sorted(test_f):
        refs = [np.asarray(t, float) for t in meta[i]["rtrajs"]]; labels = np.asarray(meta[i]["rscores"], float)
        spd = float(spd_r[i]); ego = ego_r[i]; vx, vy = float(ego[0]), float(ego[1])
        steps = (np.arange(1, T + 1) * DT)[:, None]
        emean = mp[:, i].mean(0)                                  # planner default output
        d0 = emean[-1] - emean[0]; h = np.arctan2(d0[1], d0[0]) if np.linalg.norm(d0) > 1e-3 else 0.0
        perp = np.array([-np.sin(h), np.cos(h)])
        proposals = [mp[k, i] for k in range(6)]                  # the 6 multimodal samples
        proposals.append(steps * np.array([vx, vy]))             # constant velocity (plausible)
        proposals += [emean + 0.5 * perp, emean - 0.5 * perp]    # small lateral variants
        proposals += [emean[0] + (emean - emean[0]) * 0.85, emean[0] + (emean - emean[0]) * 1.15]  # speed variants
        proposals = [np.asarray(c, float)[:T] for c in proposals if np.asarray(c, float).shape[0] >= T]
        true = np.array([rfs_of(c, refs, labels, spd) for c in proposals])
        pred = reward.predict(np.array([feats(c, ego) for c in proposals]))
        rows.append({
            "default": float(rfs_of(emean, refs, labels, spd)),   # planner's own pick
            "reranked": float(true[int(np.argmax(pred))]),         # reward model's pick
            "random": float(true.mean()),                          # expected random pick
            "oracle": float(true.max()),                           # best possible in pool
            "floor": float(true.min()),
        })
    R = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    print(f"\nG3 over {len(rows)} held-out scenes — mean TRUE RFS by selection strategy:", flush=True)
    for k in ("floor", "random", "default", "reranked", "oracle"):
        print(f"  {k:9s} {R[k].mean():.3f}", flush=True)

    diff = R["reranked"] - R["default"]
    uq = np.arange(len(diff))
    boots = [diff[RNG.choice(uq, len(uq), replace=True)].mean() for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    v = "LIFT (CI excl 0)" if lo > 0 else ("worse" if hi < 0 else "inconclusive")
    print(f"\n[reranked - default] Δ={diff.mean():+.3f} RFS  95% CI[{lo:+.3f},{hi:+.3f}] -> {v}", flush=True)
    dr = R["reranked"] - R["random"]
    boots2 = [dr[RNG.choice(uq, len(uq), replace=True)].mean() for _ in range(5000)]
    lo2, hi2 = np.percentile(boots2, [2.5, 97.5])
    print(f"[reranked - random]  Δ={dr.mean():+.3f} RFS  95% CI[{lo2:+.3f},{hi2:+.3f}]", flush=True)
    capt = (R["reranked"].mean() - R["random"].mean()) / (R["oracle"].mean() - R["random"].mean() + 1e-9)
    print(f"[oracle gap captured] {100 * capt:.1f}%  (random->oracle headroom)", flush=True)
    json.dump({k: float(R[k].mean()) for k in R} | {"lift_default_ci": [float(lo), float(hi)],
              "lift_random_ci": [float(lo2), float(hi2)], "n_scenes": len(rows)},
              open(os.path.expanduser("~/g3_out.json"), "w"))
    print("WROTE ~/g3_out.json", flush=True)


if __name__ == "__main__":
    main()
