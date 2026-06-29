"""Phase 2 / Phase A — G1 DE-RISK: validate on REAL planner trajectories (no synthetic perturbations).

The probe used perturbations of the human/rater trajectories, which could leak perturbation
signatures. Here the candidate pool is genuine planner outputs + driving heuristics:
  - a trained ego-status MLP ensemble (K=6 seeds) predicting the future from ego state
  - constant-velocity (at 0.5x / 1.0x / 1.5x current speed)
  - a stopped trajectory
  - the mean of the rater-preferred trajectories
  - the global-mean training future (a context-free prior)
  - pairwise blends of two ensemble members
None of these is the probe's perturbation scheme. We score each candidate's official RFS and
test whether a geometry+ego model predicts / ranks RFS on HELD-OUT scenes. Two tests:
  (A) train+test on real candidates (held-out scenes) -> the honest G1 number
  (B) train on the probe's PERTURBATIONS, test on these REAL candidates -> cross-distribution transfer
Baseline throughout: ADE-to-human oracle (uses the label our model never sees).
"""
# ruff: noqa: E702  (compact exploratory probe scripts)
import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from waymo_open_dataset.metrics.python import rater_feedback_utils as rfu

CACHE = os.environ.get("CACHE", "/home/dev_alfred_ai_app/wod/wod_surround_emb.npz")
T, DT = 20, 0.25
RNG = np.random.RandomState(17)


def rfs_of(cand, refs, labels, spd):
    out = rfu.get_rater_feedback_score(cand.reshape(1, 1, T, 2), np.array([[1.0]]),
                                       [refs], [labels], np.array([spd]), frequency=4, length_seconds=5)
    return float(np.array(out["rater_feedback_score"]).ravel()[0])


def feats(cand, ego):
    diffs = np.diff(cand, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    head = np.arctan2((cand[-1] - cand[0])[1], (cand[-1] - cand[0])[0])
    perp = np.array([-np.sin(head), np.cos(head)])
    lat = (cand - cand[0]) @ perp
    ang = np.arctan2(diffs[:, 1], diffs[:, 0])
    g = [float(seg.sum()), float(np.linalg.norm(cand[-1] - cand[0])), float(cand[-1, 0]), float(cand[-1, 1]),
         float(np.abs(lat).max()), float(lat.mean()), float(seg.mean()), float(seg.std()),
         float(np.abs(np.diff(ang)).sum()), float(seg.max()), float(seg.min())]
    return np.concatenate([cand.reshape(-1), np.array(g, float), np.array(ego, float)])


def perturb(traj):  # the PROBE scheme (for cross-distribution training only)
    out = [traj]
    d = traj[-1] - traj[0]
    h = np.arctan2(d[1], d[0]) if np.linalg.norm(d) > 1e-3 else 0.0
    perp = np.array([-np.sin(h), np.cos(h)])
    for sh in (-2., -1., -.5, .5, 1., 2.):
        out.append(traj + sh * perp)
    for sc in (.7, .85, 1.15, 1.3):
        out.append(traj[0] + (traj - traj[0]) * sc)
    for deg in (-10., -6., 6., 10.):
        a = np.deg2rad(deg)
        out.append(traj @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]).T)
    return out


def main():
    d = np.load(CACHE)
    ego_tr, Y_tr = d["ego_tr"], d["Y"]
    ego_r, spd_r, shard_r = d["ego_r"], d["init_speed_r"], d["shard_r"]
    meta = json.load(open(CACHE + ".rater.json"))
    n = len(meta)
    gmean_future = Y_tr.mean(0).reshape(T, 2)
    print(f"train={len(ego_tr)} rater={n}", flush=True)

    # real ego-status MLP ensemble (K=6 seeds) -> genuine planner trajectories
    sc = StandardScaler().fit(ego_tr)
    Xs = sc.transform(ego_tr)
    members = []
    for k in range(6):
        m = MLPRegressor(hidden_layer_sizes=(128, 128), alpha=1e-4, max_iter=300, random_state=100 + k)
        m.fit(Xs, Y_tr)
        members.append(m)
    print("trained 6-member ego ensemble", flush=True)
    egr = sc.transform(ego_r)
    member_pred = np.stack([m.predict(egr).reshape(n, T, 2) for m in members], 0)  # [6,n,T,2]

    def real_candidates(i):
        ego = ego_r[i]
        vx, vy = float(ego[0]), float(ego[1])
        steps = (np.arange(1, T + 1) * DT)[:, None]
        cands = [member_pred[k, i] for k in range(6)]                     # real learned planner
        for s in (0.5, 1.0, 1.5):                                          # constant velocity
            cands.append(steps * np.array([vx, vy]) * s)
        cands.append(np.zeros((T, 2)) + 1e-3)                             # stopped
        cands.append(np.mean([np.asarray(t, float) for t in meta[i]["rtrajs"]], 0))  # mean rater
        cands.append(gmean_future.copy())                                 # context-free prior
        a, b = member_pred[0, i], member_pred[3, i]                       # blend of two members
        cands.append(0.5 * a + 0.5 * b)
        return cands

    def build(cand_fn):
        X, Y, fid, sh, ade = [], [], [], [], []
        for i in range(n):
            refs = [np.asarray(t, float) for t in meta[i]["rtrajs"]]
            labels = np.asarray(meta[i]["rscores"], float)
            fut = np.asarray(meta[i]["future"], float)
            spd = float(spd_r[i])
            for c in cand_fn(i):
                c = np.asarray(c, float)[:T]
                if c.shape[0] < T:
                    continue
                X.append(feats(c, ego_r[i])); Y.append(rfs_of(c, refs, labels, spd))
                fid.append(i); sh.append(int(shard_r[i]))
                ade.append(-float(np.mean(np.linalg.norm(c - fut, axis=1))))
        return (np.array(X), np.array(Y), np.array(fid), np.array(sh), np.array(ade))

    Xr, Yr, fidr, shr, ader = build(real_candidates)
    print(f"REAL candidates: samples={len(Xr)} per_scene~{len(Xr)//n} RFS mean={Yr.mean():.2f} "
          f"sd={Yr.std():.2f} range[{Yr.min():.1f},{Yr.max():.1f}]", flush=True)

    uf = np.unique(fidr); RNG.shuffle(uf)
    test_f = set(uf[:int(0.3 * len(uf))].tolist())
    te = np.array([f in test_f for f in fidr]); trm = ~te

    def within(score_full, y, fid):
        rs = []
        for f in test_f:
            m = fid == f
            if m.sum() >= 5 and len(set(np.round(y[m], 3))) > 1:
                r = spearmanr(score_full[m], y[m]).correlation
                if not np.isnan(r):
                    rs.append(r)
        return float(np.mean(rs)), len(rs)

    # (A) train+test on REAL (held-out scenes)
    mA = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=6, random_state=17)
    mA.fit(Xr[trm], Yr[trm])
    pA = mA.predict(Xr[te])
    rhoA = spearmanr(pA, Yr[te]).correlation
    rho_or = spearmanr(ader[te], Yr[te]).correlation
    pf = np.full(len(Yr), np.nan); pf[te] = pA
    wA, nsc = within(pf, Yr, fidr)
    w_or, _ = within(ader, Yr, fidr)
    print(f"\n[A: REAL train+test] held-out Spearman(pred,RFS) = {rhoA:+.3f}  | ADE oracle = {rho_or:+.3f}", flush=True)
    print(f"[A: within-scene]     learned = {wA:+.3f}  | ADE oracle = {w_or:+.3f}  ({nsc} scenes)", flush=True)

    # (B) cross-distribution: train on PERTURBATIONS, test on REAL held-out scenes
    Xp, Yp = [], []
    for i in range(n):
        if i in test_f:
            continue  # keep test scenes unseen even in pert training
        refs = [np.asarray(t, float) for t in meta[i]["rtrajs"]]
        labels = np.asarray(meta[i]["rscores"], float)
        fut = np.asarray(meta[i]["future"], float); spd = float(spd_r[i])
        for base in [fut] + refs:
            for c in perturb(base):
                c = np.asarray(c, float)[:T]
                if c.shape[0] == T:
                    Xp.append(feats(c, ego_r[i])); Yp.append(rfs_of(c, refs, labels, spd))
    Xp, Yp = np.array(Xp), np.array(Yp)
    mB = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=6, random_state=17)
    mB.fit(Xp, Yp)
    pB = mB.predict(Xr[te])
    rhoB = spearmanr(pB, Yr[te]).correlation
    pfB = np.full(len(Yr), np.nan); pfB[te] = pB
    wB, _ = within(pfB, Yr, fidr)
    print(f"\n[B: train PERTURB -> test REAL] held-out Spearman = {rhoB:+.3f}  within-scene = {wB:+.3f}", flush=True)

    json.dump({"realA_rho": float(rhoA), "realA_within": float(wA), "ade_rho": float(rho_or),
               "ade_within": float(w_or), "crossB_rho": float(rhoB), "crossB_within": float(wB),
               "n_real": int(len(Xr)), "n_scenes": int(nsc)},
              open(os.path.expanduser("~/derisk_g1_out.json"), "w"))
    print("\nWROTE ~/derisk_g1_out.json", flush=True)


if __name__ == "__main__":
    main()
