"""Phase 2 / Phase A — does SCENE PERCEPTION clear G1?

Geometry+ego plateaued at the ADE oracle (0.573 vs 0.603) on real planner trajectories.
The only unused information is what is in the scene. We add the cached DINOv2 8-camera
embeddings (already on disk, no GPU) as scene context and re-test on REAL planner
trajectories, held-out by scene. Reports: geom+ego vs geom+ego+scene vs ADE oracle, with a
drive-cluster bootstrap CI on the (scene - no-scene) held-out Spearman difference.
"""
# ruff: noqa: E702  (compact exploratory probe scripts)
import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from waymo_open_dataset.metrics.python import rater_feedback_utils as rfu

CACHE = os.environ.get("CACHE", "/home/dev_alfred_ai_app/wod/wod_surround_emb.npz")
T, DT, PCA_D = 20, 0.25, 96
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
    ego_tr, Y_tr, _emb_tr = d["ego_tr"], d["Y"], d["emb_tr"]
    ego_r, spd_r, _shard_r, emb_r = d["ego_r"], d["init_speed_r"], d["shard_r"], d["emb_r"]
    meta = json.load(open(CACHE + ".rater.json"))
    n = len(meta)
    gmean = Y_tr.mean(0).reshape(T, 2)

    sc = StandardScaler().fit(ego_tr); Xs = sc.transform(ego_tr)
    members = [MLPRegressor(hidden_layer_sizes=(128, 128), alpha=1e-4, max_iter=300, random_state=100 + k).fit(Xs, Y_tr)
               for k in range(6)]
    mp = np.stack([m.predict(sc.transform(ego_r)).reshape(n, T, 2) for m in members], 0)
    print(f"train={len(ego_tr)} rater={n}; ego ensemble ready", flush=True)

    # split scenes first so PCA is fit on train scenes only
    uf = np.arange(n); RNG.shuffle(uf)
    test_f = set(uf[:int(0.3 * n)].tolist())
    pca = PCA(PCA_D, random_state=17).fit(emb_r[[i for i in range(n) if i not in test_f]].reshape(-1, emb_r.shape[1] * emb_r.shape[2]))
    scene_pca = pca.transform(emb_r.reshape(n, -1))  # [n, PCA_D]
    print(f"scene PCA({PCA_D}) evr={pca.explained_variance_ratio_.sum():.3f}", flush=True)

    Xg, Xs2, Yr, fid, ade = [], [], [], [], []
    for i in range(n):
        refs = [np.asarray(t, float) for t in meta[i]["rtrajs"]]; labels = np.asarray(meta[i]["rscores"], float)
        fut = np.asarray(meta[i]["future"], float); spd = float(spd_r[i]); ego = ego_r[i]
        vx, vy = float(ego[0]), float(ego[1]); steps = (np.arange(1, T + 1) * DT)[:, None]
        cands = [mp[k, i] for k in range(6)]
        cands += [steps * np.array([vx, vy]) * s for s in (0.5, 1.0, 1.5)]
        cands += [np.zeros((T, 2)) + 1e-3, np.mean(refs, 0), gmean.copy(), 0.5 * mp[0, i] + 0.5 * mp[3, i]]
        for c in cands:
            c = np.asarray(c, float)[:T]
            if c.shape[0] < T:
                continue
            gv = feats(c, ego)
            Xg.append(gv); Xs2.append(np.concatenate([gv, scene_pca[i]]))
            Yr.append(rfs_of(c, refs, labels, spd)); fid.append(i)
            ade.append(-float(np.mean(np.linalg.norm(c - fut, axis=1))))
    Xg, Xs2, Yr, fid, ade = map(np.array, (Xg, Xs2, Yr, fid, ade))
    te = np.array([f in test_f for f in fid]); tr = ~te
    print(f"real samples={len(Xg)} RFS sd={Yr.std():.2f}", flush=True)

    def fit_eval(X):
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=6, random_state=17)
        m.fit(X[tr], Yr[tr]); return m.predict(X[te])

    pg, ps = fit_eval(Xg), fit_eval(Xs2)
    yte, fte, adte = Yr[te], fid[te], ade[te]
    rg, rs2, ro = (spearmanr(p, yte).correlation for p in (pg, ps, adte))
    print(f"\n[held-out Spearman vs RFS] geom+ego={rg:+.3f}  +scene={rs2:+.3f}  ADE-oracle={ro:+.3f}", flush=True)

    def within(score):
        out = []
        for f in set(fte.tolist()):
            m = fte == f
            if m.sum() >= 5 and len(set(np.round(yte[m], 3))) > 1:
                r = spearmanr(score[m], yte[m]).correlation
                if not np.isnan(r):
                    out.append(r)
        return float(np.mean(out))
    print(f"[within-scene]            geom+ego={within(pg):+.3f}  +scene={within(ps):+.3f}  ADE-oracle={within(adte):+.3f}", flush=True)

    # cluster-bootstrap CI on (scene - geom) held-out Spearman difference, by scene
    uq = np.unique(fte); idx = {f: np.where(fte == f)[0] for f in uq}
    diffs = []
    for _ in range(2000):
        rows = np.concatenate([idx[f] for f in RNG.choice(uq, len(uq), replace=True)])
        diffs.append(spearmanr(ps[rows], yte[rows]).correlation - spearmanr(pg[rows], yte[rows]).correlation)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    v = "scene HELPS (CI excl 0)" if lo > 0 else ("scene HURTS" if hi < 0 else "inconclusive")
    print(f"\n[scene - geom] Δrho={rs2 - rg:+.3f}  95% CI[{lo:+.3f},{hi:+.3f}] -> {v}", flush=True)
    print(f"[G1 bar 0.60] geom+ego {'PASS' if rg >= 0.6 else 'miss'} | +scene {'PASS' if rs2 >= 0.6 else 'miss'}", flush=True)
    json.dump({"geom": float(rg), "scene": float(rs2), "ade": float(ro),
               "delta_ci": [float(lo), float(hi)]}, open(os.path.expanduser("~/derisk_scene_out.json"), "w"))
    print("WROTE ~/derisk_scene_out.json", flush=True)


if __name__ == "__main__":
    main()
