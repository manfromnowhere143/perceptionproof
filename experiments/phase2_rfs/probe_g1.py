"""Phase 2 / Phase A — G1 feasibility probe (CPU, no GPU, no re-parse).

Question: is the human Rater Feedback Score predictable from a candidate trajectory's own
geometry + ego context, WITHOUT the human reference trajectories? If yes, a supervised
RFS predictor is feasible and scene context (GPU) only helps. If no, we escalate.

Method: for each of the 479 rater frames (cached references + scores + ego), generate a
diverse candidate set (human future + rater trajectories + lateral/speed/rotation
perturbations), score each candidate with the OFFICIAL RFS, then train a model to predict
RFS from candidate geometry + ego (no references seen). Evaluate on HELD-OUT scenes:
  - overall held-out Spearman rho(pred, true RFS), drive-cluster bootstrap
  - mean WITHIN-scene ranking quality (Spearman over each held-out scene's candidates) —
    this is what a reranker (G3) actually needs
Baselines: predict-mean; ADE-to-human ORACLE (ranks candidates by closeness to the human
future — needs the label our model never sees).
"""
# ruff: noqa: E702  (compact exploratory probe scripts)
import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from waymo_open_dataset.metrics.python import rater_feedback_utils as rfu

CACHE = os.environ.get("CACHE", "/home/dev_alfred_ai_app/wod/wod_surround_emb.npz")
T = 20
RNG = np.random.RandomState(17)


def rfs_of(cand, refs, labels, spd):
    out = rfu.get_rater_feedback_score(cand.reshape(1, 1, T, 2), np.array([[1.0]]),
                                       [refs], [labels], np.array([spd]),
                                       frequency=4, length_seconds=5)
    return float(np.array(out["rater_feedback_score"]).ravel()[0])


def perturb(traj):
    """Yield geometric variants of a [T,2] trajectory to span a range of RFS."""
    out = [traj]
    d = traj[-1] - traj[0]
    heading = np.arctan2(d[1], d[0]) if np.linalg.norm(d) > 1e-3 else 0.0
    perp = np.array([-np.sin(heading), np.cos(heading)])
    for sh in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0):       # lateral shift (m)
        out.append(traj + sh * perp)
    for sc in (0.7, 0.85, 1.15, 1.3):                  # speed / extent scale
        out.append(traj[0] + (traj - traj[0]) * sc)
    for deg in (-10.0, -6.0, 6.0, 10.0):               # rotation about ego origin
        a = np.deg2rad(deg)
        rmat = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        out.append(traj @ rmat.T)
    return out


def feats(cand, ego):
    """Geometry of the candidate (NO references) + ego context."""
    diffs = np.diff(cand, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    arc = float(seg.sum())
    net = float(np.linalg.norm(cand[-1] - cand[0]))
    head = np.arctan2((cand[-1] - cand[0])[1], (cand[-1] - cand[0])[0])
    perp = np.array([-np.sin(head), np.cos(head)])
    lat = (cand - cand[0]) @ perp
    ang = np.arctan2(diffs[:, 1], diffs[:, 0])
    curv = float(np.abs(np.diff(ang)).sum())
    g = [arc, net, float(cand[-1, 0]), float(cand[-1, 1]), float(np.abs(lat).max()),
         float(lat.mean()), float(seg.mean()), float(seg.std()), curv,
         float(seg.max()), float(seg.min())]
    return np.concatenate([cand.reshape(-1), np.array(g, float), np.array(ego, float)])


def main():
    d = np.load(CACHE)
    ego_r, spd_r, shard_r = d["ego_r"], d["init_speed_r"], d["shard_r"]
    meta = json.load(open(CACHE + ".rater.json"))
    n = len(meta)
    print(f"rater frames={n}", flush=True)

    X, Y, frame_id, shard_id, ade_oracle = [], [], [], [], []
    for i in range(n):
        refs = [np.asarray(t, float) for t in meta[i]["rtrajs"]]
        labels = np.asarray(meta[i]["rscores"], float)
        fut = np.asarray(meta[i]["future"], float)
        spd = float(spd_r[i])
        cands = []
        for base in [fut] + refs:
            cands += perturb(base)
        for c in cands:
            c = np.asarray(c, float)[:T]
            if c.shape[0] < T:
                continue
            X.append(feats(c, ego_r[i]))
            Y.append(rfs_of(c, refs, labels, spd))
            frame_id.append(i)
            shard_id.append(int(shard_r[i]))
            ade_oracle.append(-float(np.mean(np.linalg.norm(c - fut, axis=1))))  # -ADE: higher=better
        if i % 80 == 0:
            print(f"  frame {i}: samples={len(X)}", flush=True)
    X = np.array(X); Y = np.array(Y); frame_id = np.array(frame_id)
    shard_id = np.array(shard_id); ade_oracle = np.array(ade_oracle)
    print(f"samples={len(X)} feat_dim={X.shape[1]} RFS mean={Y.mean():.2f} sd={Y.std():.2f} "
          f"min={Y.min():.1f} max={Y.max():.1f}", flush=True)

    # held-out split by FRAME (generalize to unseen scenes), 70/30
    uf = np.unique(frame_id)
    RNG.shuffle(uf)
    test_f = set(uf[:int(0.3 * len(uf))].tolist())
    te = np.array([f in test_f for f in frame_id])
    tr = ~te
    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                          max_depth=6, random_state=17)
    model.fit(X[tr], Y[tr])
    pred = model.predict(X[te])
    yte = Y[te]

    rho = spearmanr(pred, yte).correlation
    rho_oracle = spearmanr(ade_oracle[te], yte).correlation
    print(f"\n[G1] held-out per-candidate Spearman(pred, RFS)   = {rho:+.3f}", flush=True)
    print(f"[baseline] ADE-to-human ORACLE Spearman           = {rho_oracle:+.3f} (uses the label)", flush=True)
    print("[baseline] predict-mean R^2 ~ 0 by construction", flush=True)

    # within-scene ranking (what a reranker needs): per held-out scene, Spearman of the
    # scoring vs true RFS over that scene's candidates, averaged over scenes.
    pred_full = np.full(len(Y), np.nan)
    pred_full[te] = pred

    def within(score_full):
        rs = []
        for f in test_f:
            m = frame_id == f
            if m.sum() >= 5 and len(set(np.round(Y[m], 3))) > 1:
                r = spearmanr(score_full[m], Y[m]).correlation
                if not np.isnan(r):
                    rs.append(r)
        return float(np.mean(rs)), len(rs)

    w_model, nsc = within(pred_full)
    w_oracle, _ = within(ade_oracle)
    print(f"\n[G2-precursor] mean WITHIN-scene Spearman, learned model = {w_model:+.3f} (over {nsc} scenes)", flush=True)
    print(f"[G2-precursor] mean WITHIN-scene Spearman, ADE oracle    = {w_oracle:+.3f}", flush=True)

    json.dump({"rho": float(rho), "rho_oracle": float(rho_oracle),
               "within_model": float(w_model), "within_oracle": float(w_oracle),
               "n_samples": int(len(X)), "n_test_scenes": int(nsc)},
              open(os.path.expanduser("~/probe_g1_out.json"), "w"))
    print("\nWROTE ~/probe_g1_out.json", flush=True)


if __name__ == "__main__":
    main()
